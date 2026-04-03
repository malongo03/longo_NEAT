use std::collections::HashMap;
use crate::genome::{genome_distance, genome_crossover, Genome, NeuronGene, NeuronType, SynapseGene};
use std::ops::{Deref, DerefMut};
use rand::distr::Uniform;
use rand::prelude::*;

struct SpeciesHistory {
    id: usize,
    birth_gen: usize,
    origin_species: usize,
    extinct_gen: Option<usize>,
    max_fitness: Vec<f64>,
    generation_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
struct Generation(Vec<Genome>);
impl Deref for Generation {
    type Target = Vec<Genome>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Generation {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

struct History {
    reprs: Vec<Generation>,
    bests: Vec<Generation>,
    max_fitness: Vec<f64>,
    species: Vec<SpeciesHistory>,
}

// Eventually, these will all be made actual settings
const INHERIT_DISABLE_PROB: f64 = 0.75;

const MUTATE_GENE_WEIGHT: f64 = 0.8;
const MUTATE_RANDOM_WEIGHT: f64 = 0.1;
const MUTATE_ENABLE_PROB: f64 = 0.05;
const MUTATE_DISABLE_PROB: f64 = 0.005;
const MUTATION_POWER: f64 = 2.5;
const MUTATE_NEW_EDGE: f64 = 0.05;
const MUTATE_NEW_NODE: f64 = 0.03;

const CAP_WEIGHT: f64 = 8.0;
const CAP_NEW_WEIGHT: f64 = 1.0;

const EVOLUTION_TRUNCATION: f64 = 0.5;
const EVOLUTION_STAGNATION_EPSILON: f64 = 1e-5;
const EVOLUTION_NO_CROSSOVER: f64 = 0.25;
const EVOLUTION_INTERSPECIES: f64 = 0.001;

const DISTANCE_DISJOINT_WEIGHT: f64 = 1.0;
const DISTANCE_EXCESS_WEIGHT: f64 = 1.0;
const DISTANCE_WEIGHT_DIFF_WEIGHT: f64 = 3.0;

struct ActiveSpecies {
    species_history_id: usize,
    member_indices: Vec<usize>,

    last_improvement_counter: usize,
    last_improvement_fitness: f64,
}
impl ActiveSpecies {
    fn new(species_history_id: usize) -> Self {
        Self {
            species_history_id,
            member_indices: Vec::new(),

            last_improvement_counter: 0,
            last_improvement_fitness: f64::NEG_INFINITY,
        }
    }
}

/// # Interface Contract
///
/// Any Genome used as a starting network must have at least one active path from a sensory
/// neuron to a muscular neuron.
struct Population {
    population_size: usize,
    species_threshold: f64,

    active_species: Vec<ActiveSpecies>,
    population: Generation,
    fitness: Vec<Option<f64>>,

    next_inno_index: usize,
    next_node_index: usize,
    next_species_id: usize,
    generation_index: usize,

    last_improvement_counter: usize,
    last_improvement_fitness: f64,

    history: History
}
impl Population {
    pub fn new_generation(&mut self) {
        let mut rng = rand::rng();

        self.generation_index += 1;

        let num_active_species = self.active_species.len();

        // 1. Perform fitness calculations and cull species population
        let (best_fitness_per_species,
            average_fitness_per_species,
            champions,
            anchors) = self.calculate_species_fitness_and_cull();
        let mut anchors = anchors; // This has to be mutable.

        // 2. Check which species are stagnating
        let stagnating_species: Vec<bool> = self.check_for_stagnation(best_fitness_per_species);

        // 3. Calculate total valid fitness
        let total_fitness: f64 = average_fitness_per_species.iter()
            .enumerate()
            .filter(|(idx, _)| !stagnating_species[*idx])
            .map(|(_, &fitness)| fitness)
            .sum();

        // 4. Determine allotments for each active species
        let mut remainder: usize = self.population_size;
        let mut allotted_children: Vec<usize> = vec![0; num_active_species];
        for (active_index, avg_fitness) in average_fitness_per_species.into_iter().enumerate() {
            if stagnating_species[active_index] {
                continue;
            }
            let allotment: usize = (avg_fitness / total_fitness).floor() as usize;
            allotted_children[active_index] = allotment;
            remainder -= allotment;
        }
        // 4.5. Make sure that the full population size was
        while remainder > 0 {
            for index in (0..num_active_species).rev() {
                if !stagnating_species[index] {
                    allotted_children[index] += 1;
                    remainder -= 1;
                }
                if remainder == 0 {
                    break;
                }
            }
        }

        // 5. Produce new generation, sorted into proper species
        let mut new_indices: Vec<Vec<usize>> = Vec::with_capacity(num_active_species);
        for &num_children in allotted_children.iter() {
            new_indices.push(Vec::with_capacity(num_children));
        }
        let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
        let mut new_nodes: HashMap<usize, usize> = HashMap::new();
        let mut new_generation = Generation(Vec::with_capacity(num_active_species));
        for (active_index, mut num_children) in allotted_children.into_iter().enumerate() {
            let num_potential_parents: usize = self.active_species[active_index].member_indices.len();

            // Clone best performing genome of the species from last generation if there is room.
            if num_children >= 5 {
                let champion = &champions[active_index];

                self.sort_and_place_new_genome(&mut new_generation,
                                               &mut new_indices,
                                               &mut anchors,
                                               champion.neuron_genes().clone(),
                                               champion.synapse_genes().clone(),
                                               champion.species_history_id());

                num_children -= 1;
            }

            for _ in 0..num_children {
                let parent1_index = *self.active_species[active_index].member_indices
                    .choose(&mut rng).expect("Pop indices should always be non-empty");
                let parent1: &Genome = &self.population[parent1_index];
                let fitness1: f64 = self.fitness[parent1_index].expect(
                    "The fitness vector should already have been sanitized."
                );

                let (neuron_genes, synapse_genes, origin_species) =

                    // No Crossover
                    if rng.random_bool(EVOLUTION_NO_CROSSOVER) {
                        let genome = &self.population[parent1_index];
                        (genome.neuron_genes().clone(), genome.synapse_genes().clone(), genome.species_history_id())
                    }

                    // Interspecies Crossover
                    else if rng.random_bool(EVOLUTION_INTERSPECIES) {
                        if num_active_species > 1 {
                            let species_index = rng.random_range(0..num_active_species - 1);
                            let species_index =
                                if active_index <= species_index {
                                    species_index + 1
                                } else {
                                    species_index
                                };
                            let parent2_index = *self.active_species[species_index].member_indices
                                .choose(&mut rng).expect("Pop indices will always be non-empty");
                            let parent2: &Genome = &self.population[parent2_index];
                            let fitness2: f64 = self.fitness[parent2_index].expect(
                                "The fitness vector should already have been sanitized."
                            );

                            genome_crossover(parent1, parent2, fitness1, fitness2, INHERIT_DISABLE_PROB)
                        }
                        else {
                            (parent1.neuron_genes().clone(), parent1.synapse_genes().clone(), parent1.species_history_id())
                        }
                    }

                    // Normal Crossover
                    else {
                        'found_index: {
                            if num_potential_parents > 1 {
                                for _ in 0..20 {
                                    let parent2_index = *self.active_species[active_index].member_indices
                                        .choose(&mut rng).expect("Pop indices will always be non-empty");
                                    if parent1_index == parent2_index {
                                        continue;
                                    }
                                    let parent2: &Genome = &self.population[parent2_index];
                                    let fitness2: f64 = self.fitness[parent2_index].expect(
                                        "The fitness vector should already have been sanitized."
                                    );

                                    break 'found_index genome_crossover(parent1, parent2, fitness1, fitness2, INHERIT_DISABLE_PROB);
                                }
                            }
                            (parent1.neuron_genes().clone(), parent1.synapse_genes().clone(), parent1.species_history_id())
                        }
                    };
                let (neuron_genes, synapse_genes) = self.mutate(neuron_genes, synapse_genes, &mut new_innos, &mut new_nodes);

                self.sort_and_place_new_genome(&mut new_generation,
                                               &mut new_indices,
                                               &mut anchors,
                                               neuron_genes,
                                               synapse_genes,
                                               origin_species);
            }
        }

        // 6. Perform extinctions
        let mut new_active_species: Vec<ActiveSpecies> = Vec::with_capacity(num_active_species);
        for (active_index, member_indices) in new_indices.into_iter().enumerate() {
            let old_species = &self.active_species[active_index];
            if member_indices.len() == 0 {
                self.history.species[old_species.species_history_id].extinct_gen = Some(self.generation_index);
                continue;
            }
            new_active_species.push(ActiveSpecies{
                species_history_id: old_species.species_history_id,
                member_indices,
                last_improvement_counter: old_species.last_improvement_counter,
                last_improvement_fitness: old_species.last_improvement_fitness,
            });
        }

        // 7. Move everything
        self.population = new_generation;
        self.active_species = new_active_species;
        for fitness in self.fitness.iter_mut() {
            *fitness = None;
        }
    }

    fn calculate_species_fitness_and_cull(&mut self) -> (Vec<f64>, Vec<f64>, Generation, Generation) {
        // 0.) Sanitize fitness scores. We default to a fitness of 0.0 if no fitness has been assigned.
        for fitness in self.fitness.iter_mut() {
            fitness.get_or_insert(0.0);
        }

        let mut rng = rand::rng();

        let num_active_species: usize = self.active_species.len();

        let mut best_fitness_per_species: Vec<f64> = Vec::with_capacity(num_active_species);
        let mut average_fitness_per_species: Vec<f64> = Vec::with_capacity(num_active_species);
        let mut champions: Generation = Generation(Vec::with_capacity(num_active_species));
        let mut anchors: Generation = Generation(Vec::with_capacity(num_active_species));

        // 1.) Find a champion for each active species
        // 2.) Find a random representative for each active species
        // 3.) Find the average fitness for each active species
        // 4.) Truncate each species by killing a fraction of worst scorers
        // 5.) Save the best fitness to history
        for (active_index, active_species) in self.active_species.iter_mut().enumerate() {
            // 1
            active_species.member_indices.sort_unstable_by(|a, b| {
                let fit_a = self.fitness[*a].expect("Already sanitized");
                let fit_b = self.fitness[*b].expect("Already sanitized");
                fit_b.partial_cmp(&fit_a).expect("Fitness cannot be NaN")
            });
            let champion_index: usize = active_species.member_indices[0];
            let champion_fitness: f64 = self.fitness[champion_index].expect("Already sanitized");
            champions.push(self.population[champion_index].clone());
            best_fitness_per_species.push(champion_fitness);

            // 2
            let repr_index: usize = *active_species.member_indices.choose(&mut rng).expect(
                "An empty active species should have been made extinct and filtered out."
            );
            anchors.push(self.population[repr_index].clone());

            // 3
            let mut species_fitness = 0.0;
            let species_pop: f64 = active_species.member_indices.len() as f64;
            for &genome_index in active_species.member_indices.iter() {
                let fitness = self.fitness[genome_index].expect("Already sanitized");
                let adj_fitness = fitness / species_pop;
                species_fitness += adj_fitness;
            }
            average_fitness_per_species.push(species_fitness);

            // 4
            let survival_count = active_species.member_indices.len() - (EVOLUTION_TRUNCATION * species_pop).floor() as usize;
            active_species.member_indices.truncate(survival_count);

            // 5
            self.history.species[active_species.species_history_id].max_fitness.push(champion_fitness);
            self.history.species[active_species.species_history_id].generation_indices.push(active_index);
        }

        // 6. Save current generation's genomes for historical analysis
        self.history.reprs.push(anchors.clone());
        self.history.bests.push(champions.clone());

        (best_fitness_per_species, average_fitness_per_species, champions, anchors)
    }

    fn check_for_stagnation(&mut self, best_fitness_per_species: Vec<f64>) -> Vec<bool> {
        let num_active_species = self.active_species.len();

        let mut stagnating: Vec<bool> = vec![false; num_active_species];
        if num_active_species == 1 { // No population or species stagnation can occur during zero diversity

            let top1_score: f64 = best_fitness_per_species[0];

            self.history.max_fitness.push(top1_score);

            self.last_improvement_counter = 0;
            self.last_improvement_fitness = top1_score;

            let active_species = &mut self.active_species[0];

            let fitness = best_fitness_per_species[0];
            active_species.last_improvement_counter += 1;
            if fitness - active_species.last_improvement_fitness > EVOLUTION_STAGNATION_EPSILON {
                active_species.last_improvement_counter = 0;
                active_species.last_improvement_fitness = fitness;
            }
            return stagnating
        }

        // 1.) Find top two scoring species
        let mut top1_score: f64 = f64::NEG_INFINITY;
        let mut top1_index: usize = 0;
        let mut top2_score: f64 = f64::NEG_INFINITY;
        let mut top2_index: usize = 1;
        for (index, &fitness) in best_fitness_per_species.iter().enumerate() {
            if fitness > top1_score {
                top2_score = top1_score;
                top2_index = top1_index;

                top1_score = fitness;
                top1_index = index;
            }
            else if fitness > top2_score {
                top2_score = fitness;
                top2_index = index;
            }
        }

        // To be honest, this max fitness recording should be in the main next_generation function
        // and not in here.
        self.history.max_fitness.push(top1_score);

        // 2.) Check if the entire population is stagnating. If so, all but the top two species
        // will be alloted zero children.
        let population_stagnation: bool = {
            self.last_improvement_counter += 1;
            if top1_score - self.last_improvement_fitness > EVOLUTION_STAGNATION_EPSILON {
                self.last_improvement_counter = 0;
                self.last_improvement_fitness = top1_score;
                false
            }
            else if self.last_improvement_counter >= 20 {
                self.last_improvement_counter = 0;
                self.last_improvement_fitness = top1_score;
                true
            }
            else {
                false
            }
        };
        if population_stagnation {
            for (index, stagnation) in stagnating.iter_mut().enumerate() {
                // Elitism
                if index != top1_index && index != top2_index {
                    *stagnation = true;
                }
            }
        }

        // 3.) Check for species stagnation (Note: this still has to run even if the whole
        // population is stagnating. This is because species which are marked stagnant are not
        // guaranteed to go extinct.)
        for (active_index, active_species) in self.active_species.iter_mut().enumerate() {
            let fitness = best_fitness_per_species[active_index];
            active_species.last_improvement_counter += 1;
            if fitness - active_species.last_improvement_fitness > EVOLUTION_STAGNATION_EPSILON {
                active_species.last_improvement_counter = 0;
                active_species.last_improvement_fitness = fitness;
                continue;
            }
            // Elitism protection. Top two species can never stagnate.
            if active_index == top1_index || active_index == top2_index {
                continue;
            }
            if active_species.last_improvement_counter >= 15 {
                stagnating[active_index] = true;
            }
        }

        stagnating
    }

    fn mutate(&mut self,
              mut neuron_genes: Vec<NeuronGene>,
              mut synapse_genes: Vec<SynapseGene>,
              new_innos: &mut HashMap<(usize, usize), usize>,
              new_nodes: &mut HashMap<usize, usize>) -> (Vec<NeuronGene>, Vec<SynapseGene>) {
        let mut rng = rand::rng();
        let weight_dist = Uniform::new_inclusive(-MUTATION_POWER, MUTATION_POWER)
            .expect("Mutation power should be strictly positive and finite.");

        // Mutate weights on individual synapse genes.
        let num_genes = synapse_genes.len() as f64;
        for gene in synapse_genes.iter_mut() {
            if rng.random_bool(MUTATE_GENE_WEIGHT) {
                let nudge = weight_dist.sample(&mut rng);
                if rng.random_bool(MUTATE_RANDOM_WEIGHT) {
                    gene.weight = nudge;
                }
                else {
                    gene.weight += nudge;
                    gene.weight = gene.weight.clamp(-CAP_WEIGHT, CAP_WEIGHT);
                }
            }

            if gene.enabled {
                if rng.random_bool(MUTATE_DISABLE_PROB / num_genes) {
                    gene.enabled = false;
                }
            }
            else {
                if rng.random_bool(MUTATE_ENABLE_PROB / num_genes) {
                    gene.enabled = true;
                }
            }
        }

        // Mutate a new edge. Note that this is done with rejection sampling, and so it is not
        // guaranteed to succeed.
        if rng.random_bool(MUTATE_NEW_EDGE) {
            for _ in 0..20 {
                let src_id: usize = rng.random_range(0..neuron_genes.len());
                let tgt_id: usize = rng.random_range(0..neuron_genes.len());

                // Reject if the target is a sensory type neuron.
                if matches!(neuron_genes[tgt_id].neuron_type, NeuronType::Sensory(_)) {
                    continue;
                }

                let src_id: usize = neuron_genes[src_id].node_name;
                let tgt_id: usize = neuron_genes[tgt_id].node_name;

                // Reject if this edge appears elsewhere.
                if synapse_genes.iter().any(
                    |gene| gene.src_id == src_id && gene.tgt_id == tgt_id
                ) {
                    continue;
                }

                let inno_num = *new_innos.entry((src_id, tgt_id))
                    .or_insert_with(|| {
                        let new_num = self.next_inno_index;
                        self.next_inno_index += 1;
                        new_num
                    });

                let weight: f64 = rng.random_range(-CAP_NEW_WEIGHT..=CAP_NEW_WEIGHT);
                let new_synapse: SynapseGene = SynapseGene{
                    src_id,
                    tgt_id,
                    weight,
                    inno_num,
                    enabled: true,
                };
                synapse_genes.push(new_synapse);
                break;
            }
        }

        // Mutate a new node. This one is guaranteed to succeed.
        if rng.random_bool(MUTATE_NEW_NODE) {
            let target_idx = synapse_genes.iter()
                .enumerate()
                .filter(|(_, gene)| gene.enabled)
                .map(|(idx, _)| idx)
                .choose(&mut rng);

            if let Some(target_idx) = target_idx {
                let old_synapse = synapse_genes[target_idx].clone();
                synapse_genes[target_idx].enabled = false;

                let new_node_id = *new_nodes.entry(old_synapse.inno_num).or_insert_with(|| {
                    let id = self.next_node_index;
                    self.next_node_index += 1;
                    id
                });

                let is_new_split = new_node_id == self.next_node_index - 1;

                let (inno1, inno2) = if is_new_split {
                    let i1 = self.next_inno_index;
                    let i2 = self.next_inno_index + 1;
                    self.next_inno_index += 2;

                    new_innos.insert((old_synapse.src_id, new_node_id), i1);
                    new_innos.insert((new_node_id, old_synapse.tgt_id), i2);

                    (i1, i2)
                }
                else {
                    let i1 = *new_innos.get(&(old_synapse.src_id, new_node_id))
                        .expect("New node innovation already evolved.");
                    let i2 = i1 + 1;
                    (i1, i2)
                };

                synapse_genes.push(SynapseGene {
                    src_id: old_synapse.src_id,
                    tgt_id: new_node_id,
                    weight: 1.0,
                    inno_num: inno1,
                    enabled: old_synapse.enabled,
                });

                synapse_genes.push(SynapseGene {
                    src_id: new_node_id,
                    tgt_id: old_synapse.tgt_id,
                    weight: old_synapse.weight,
                    inno_num: inno2,
                    enabled: true,
                });

                neuron_genes.push(NeuronGene{
                    node_name: new_node_id,
                    neuron_type: NeuronType::Inter()
                })
            }
        }

        (neuron_genes, synapse_genes)
    }

    fn sort_and_place_new_genome(&mut self,
                                 new_generation: &mut Generation,
                                 new_indices: &mut Vec<Vec<usize>>,
                                 anchors: &mut Generation,
                                 new_neuron_genes: Vec<NeuronGene>,
                                 new_synapse_genes: Vec<SynapseGene>,
                                 origin_species: usize) {
        let id = new_generation.len();

        let species_indices: Option<(usize, usize)> = 'found_index: {
            for (active_index, anchor) in anchors.iter().enumerate() {
                let distance: f64 = genome_distance(&new_synapse_genes,
                                                    anchor.synapse_genes(),
                                                    DISTANCE_DISJOINT_WEIGHT,
                                                    DISTANCE_EXCESS_WEIGHT,
                                                    DISTANCE_WEIGHT_DIFF_WEIGHT);

                if distance <= self.species_threshold {
                    break 'found_index Some((active_index, anchor.species_history_id()))
                }
            }
            None
        };
        let (active_index, hist_index, push_anchor) =
            if let Some((active_index, hist_index)) = species_indices {
                (active_index, hist_index, false)
            }
            else {
                let active_index = self.active_species.len();
                let species_history_id = self.next_species_id;

                self.next_species_id += 1;

                new_indices.push(Vec::with_capacity(1));
                self.active_species.push(ActiveSpecies::new(species_history_id));
                self.history.species.push(SpeciesHistory{
                    id: species_history_id,
                    birth_gen: self.generation_index,
                    origin_species,
                    extinct_gen: None,
                    max_fitness: Vec::new(),
                    generation_indices: Vec::new(),
                });

                (active_index, species_history_id, true)
            };

        new_indices[active_index].push(id);

        let genome = Genome::new(id,
                                 hist_index,
                                 new_neuron_genes,
                                 new_synapse_genes)
            .expect("new_generation should not produce an invalid Genome");

        if push_anchor {
            anchors.push(genome.clone());
        }

        new_generation.push(genome);
    }

}

#[cfg(test)]
mod population_test {
    use super::*;
    use test_case::test_case;


}