use std::cmp::max;
use std::collections::HashMap;
use crate::genome::{genome_distance, Genome, NeuronGene, NeuronType, SynapseGene};
use std::ops::{Deref, DerefMut};
use std::ptr::addr_eq;
use rand::distr::Uniform;
use rand::prelude::*;
use rand::random_range;
use rand::seq::SliceRandom;

struct SpeciesHistory {
    id: usize,
    birth_gen: usize,
    origin_species: usize,
    extinct_gen: Option<usize>,
    max_fitness: Vec<f64>,
    generation_indices: Vec<usize>, // TODO
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
    historical_index: usize,
    pop_indices: Vec<usize>,

    last_improvement_counter: usize,
    last_improvement_fitness: f64,
}
impl ActiveSpecies {
    fn new(historical_index: usize) -> Self {
        Self {
            historical_index,
            pop_indices: Vec::new(),

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
    pub population_size: usize,
    pub species_threshold: f64,

    active_species: Vec<ActiveSpecies>,
    population: Generation,
    fitness: Vec<Option<f64>>,

    new_inno_index: usize,
    new_node_index: usize,
    new_species_index: usize,
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
        let mut total_fitness = 0.0;
        for (active_index, species_fitness) in average_fitness_per_species.iter().enumerate() {
            if stagnating_species[active_index] {
                continue;
            }
            total_fitness += species_fitness;
        }
        let total_fitness = total_fitness;

        // 4. Determine allotments for each active species
        let mut remainder: usize = self.population_size;
        let mut alloted_children: Vec<usize> = vec![0; num_active_species];
        for (active_index, avg_fitness) in average_fitness_per_species.into_iter().enumerate() {
            if stagnating_species[active_index] {
                continue;
            }
            let allotment: usize = (avg_fitness / total_fitness).floor() as usize;
            alloted_children[active_index] = allotment;
            remainder -= allotment;
        }
        // 4.5. Make sure that the full population size was
        while remainder > 0 {
            for index in 0..num_active_species {
                if !stagnating_species[index] {
                    alloted_children[index] += 1;
                    remainder -= 1;
                }
                if remainder == 0 {
                    break;
                }
            }
        }

        // 5. Produce new generation, sorted into proper species
        let mut new_indices: Vec<Vec<usize>> = Vec::with_capacity(num_active_species);
        for &num_children in alloted_children.iter() {
            new_indices.push(Vec::with_capacity(num_children));
        }
        let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
        let mut new_nodes: HashMap<usize, usize> = HashMap::new();
        let mut new_generation = Generation(Vec::with_capacity(num_active_species));
        for (active_index, mut num_children) in alloted_children.into_iter().enumerate() {
            let hist_index: usize = self.active_species[active_index].historical_index;
            let pot_parents: usize = self.active_species[active_index].pop_indices.len();

            // Clone best performing genome of the species from last generation if there is room.
            if num_children >= 5 {
                let champion = &champions[active_index];

                self.sort_and_place_new_genome(&mut new_generation,
                                               &mut new_indices,
                                               &mut anchors,
                                               champion.neuron_genes().clone(),
                                               champion.synapse_genes().clone(),
                                               champion.species_hist_index());

                num_children -= 1;
            }

            for _ in 0..num_children {
                let parent1_index = *self.active_species[active_index].pop_indices
                    .choose(&mut rng).expect("Pop indices will always be non-empty");
                let parent1: &Genome = &self.population[parent1_index];

                let (neuron_genes, synapse_genes, origin_index) =
                    if rng.random_bool(EVOLUTION_NO_CROSSOVER) {
                        let genome = &self.population[parent1_index];
                        (genome.neuron_genes().clone(), genome.synapse_genes().clone(), genome.species_hist_index())
                    }
                    else if !rng.random_bool(EVOLUTION_INTERSPECIES) {
                        'found_index: {
                            if pot_parents > 1 {
                                for _ in 0..20 {
                                    let parent2_index = *self.active_species[active_index].pop_indices
                                        .choose(&mut rng).expect("Pop indices will always be non-empty");
                                    if parent1_index == parent2_index {
                                        continue;
                                    }
                                    let parent2: &Genome = &self.population[parent2_index];
                                    break 'found_index self.cross_over(parent1, parent2);
                                }
                            }
                            (parent1.neuron_genes().clone(), parent1.synapse_genes().clone(), parent1.species_hist_index())
                        }
                    }
                    else {
                        'found_index: {
                            if num_active_species > 1 {
                                let species_index = rng.random_range(0..num_active_species - 1);
                                let species_index =
                                    if active_index <= species_index {
                                        species_index + 1
                                    } else {
                                        species_index
                                    };
                                let parent2_index = *self.active_species[species_index].pop_indices
                                    .choose(&mut rng).expect("Pop indices will always be non-empty");
                                let parent2: &Genome = &self.population[parent2_index];
                                break 'found_index self.cross_over(parent1, parent2);
                            }
                            (parent1.neuron_genes().clone(), parent1.synapse_genes().clone(), parent1.species_hist_index())
                        }
                    };
                let (neuron_genes, synapse_genes) = self.mutate(neuron_genes, synapse_genes, &mut new_innos, &mut new_nodes);

                self.sort_and_place_new_genome(&mut new_generation,
                                               &mut new_indices,
                                               &mut anchors,
                                               neuron_genes,
                                               synapse_genes,
                                               origin_index);
            }
        }

        // 6. Perform extinctions
        let mut new_active_species: Vec<ActiveSpecies> = Vec::with_capacity(num_active_species);
        for (active_index, pop_indices) in new_indices.into_iter().enumerate() {
            let old_species = &self.active_species[active_index];
            if pop_indices.len() == 0 {
                self.history.species[old_species.historical_index].extinct_gen = Some(self.generation_index);
                continue;
            }
            new_active_species.push(ActiveSpecies{
                historical_index: old_species.historical_index,
                pop_indices,
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
        for active_species in self.active_species.iter_mut() {
            // 1
            active_species.pop_indices.sort_unstable_by(|a, b| {
                let fit_a = self.fitness[*a].expect("Already sanitized");
                let fit_b = self.fitness[*b].expect("Already sanitized");
                fit_b.partial_cmp(&fit_a).expect("Fitness cannot be NaN")
            });
            let champion_index: usize = active_species.pop_indices[0];
            let champion_fitness: f64 = self.fitness[champion_index].expect("Already sanitized");
            champions.push(self.population[champion_index].clone());
            best_fitness_per_species.push(champion_fitness);

            // 2
            let repr_index: usize = *active_species.pop_indices.choose(&mut rng).expect(
                "An empty active species should have been made extinct and filtered out."
            );
            anchors.push(self.population[repr_index].clone());

            // 3
            let mut species_fitness = 0.0;
            let species_pop: f64 = active_species.pop_indices.len() as f64;
            for &genome_index in active_species.pop_indices.iter() {
                let fitness = self.fitness[genome_index].expect("Already sanitized");
                let adj_fitness = fitness / species_pop;
                species_fitness += adj_fitness;
            }
            average_fitness_per_species.push(species_fitness);

            // 4
            let truncate_number = active_species.pop_indices.len() - (EVOLUTION_TRUNCATION * species_pop).floor() as usize;
            active_species.pop_indices.truncate(truncate_number);

            // 5
            self.history.species[active_species.historical_index].max_fitness.push(champion_fitness);
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
            if fitness - active_species.last_improvement_fitness > EVOLUTION_TRUNCATION {
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
            if top1_score - self.last_improvement_fitness > EVOLUTION_TRUNCATION {
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
            if fitness - active_species.last_improvement_fitness > EVOLUTION_TRUNCATION {
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

    fn cross_over(&self, genome1: &Genome, genome2: &Genome) -> (Vec<NeuronGene>, Vec<SynapseGene>, usize) {
        let fitness1 = *self.fitness.get(genome1.id())
            .expect("Input genomes to cross_over should always have valid ids.");
        let fitness2 = *self.fitness.get(genome2.id())
            .expect("Input genomes to cross_over should always have valid ids.");

        // Genome with the highest fitness is considered dominant. If neither has a higher fitness,
        // we consider both Genomes dominant.
        let (dom_neur, dom_syna, dom_species, rec_neur, rec_syna, both_dom) = match (fitness1, fitness2) {
            (Some(f1), Some(f2)) => {
                if f1 > f2 {
                    (genome1.neuron_genes(),
                     genome1.synapse_genes(),
                     genome1.species_hist_index(),
                     genome2.neuron_genes(),
                     genome2.synapse_genes(),
                     false)
                }
                else if f2 > f1 {
                    (genome2.neuron_genes(),
                     genome2.synapse_genes(),
                     genome2.species_hist_index(),
                     genome1.neuron_genes(),
                     genome1.synapse_genes(),
                     false)
                }
                else {
                    (genome1.neuron_genes(),
                     genome1.synapse_genes(),
                     genome1.species_hist_index(),
                     genome2.neuron_genes(),
                     genome2.synapse_genes(),
                     true)
                }
            },
            (None, None) => {
                (genome1.neuron_genes(),
                 genome1.synapse_genes(),
                 genome1.species_hist_index(),
                 genome2.neuron_genes(),
                 genome2.synapse_genes(),
                 true)
            },
            (Some(_), None) => {
                (genome1.neuron_genes(),
                 genome1.synapse_genes(),
                 genome1.species_hist_index(),
                 genome2.neuron_genes(),
                 genome2.synapse_genes(),
                 false)
            },
            (None, Some(_)) => {
                (genome2.neuron_genes(),
                 genome2.synapse_genes(),
                 genome2.species_hist_index(),
                 genome1.neuron_genes(),
                 genome1.synapse_genes(),
                 false)
            }
        };

        let mut rng = rand::rng();

        if both_dom {
            // If both genomes are dominant, we have to inherit everything from both. (Note, this
            // should almost never happen without some kind of manual intervention.)
            let mut new_neurons: Vec<NeuronGene> = Vec::new();
            let mut new_synapses: Vec<SynapseGene> = Vec::new();

            let mut neur_iter1 = genome1.neuron_genes().iter().peekable();
            let mut neur_iter2 = genome2.neuron_genes().iter().peekable();

            while let (Some(&gene1), Some(&gene2)) = (neur_iter1.peek(), neur_iter2.peek()) {
                if gene1.node_name < gene2.node_name {
                    new_neurons.push(gene1.clone());
                    neur_iter1.next();
                }
                else if gene2.node_name < gene1.node_name {
                    new_neurons.push(gene2.clone());
                    neur_iter2.next();
                }
                else {
                    new_neurons.push(gene1.clone());
                    neur_iter1.next();
                    neur_iter2.next();
                }
            }

            let mut syna_iter1 = genome1.synapse_genes().iter().peekable();
            let mut syna_iter2 = genome2.synapse_genes().iter().peekable();

            while let (Some(&gene1), Some(&gene2)) = (syna_iter1.peek(), syna_iter2.peek()) {
                if gene1.inno_num < gene2.inno_num {
                    new_synapses.push(gene1.clone());
                    neur_iter1.next();
                }
                else if gene2.inno_num < gene1.inno_num {
                    new_synapses.push(gene2.clone());
                    neur_iter2.next();
                }
                else {
                    let mut syna_gene = if rng.random_bool(0.5) {
                        gene1.clone()
                    }
                    else {
                        gene2.clone()
                    };

                    if !gene1.enabled || !gene2.enabled {
                        syna_gene.enabled = !rng.random_bool(INHERIT_DISABLE_PROB);
                    }

                    new_synapses.push(syna_gene);
                    neur_iter1.next();
                    neur_iter2.next();
                }
            }
            while let Some(&gene) = syna_iter1.peek() {
                new_synapses.push(gene.clone());
                syna_iter1.next();
            }
            while let Some(&gene) = syna_iter2.peek() {
                new_synapses.push(gene.clone());
                syna_iter2.next();
            }

            return (new_neurons, new_synapses, dom_species);
        }

        // If one of the Genomes is dominant, then we inherit everything disjoint or excess from
        // the dominant Genome and none of these from the recessive Genome.
        let child_neurons = dom_neur.clone();
        let mut child_synapses = dom_syna.clone();

        let mut rec_iter = rec_syna.iter().peekable();

        // For any shared gene between the two Genomes, the weights are inherited from one Genome or
        // the other at a 50/50 probability. If one of the shared genes is disabled, then there is
        // a DISABLE_PROB (75%) chance of the inherited gene being disabled.
        for child_genes in child_synapses.iter_mut() {
            while let Some(&rec_gene) = rec_iter.peek() {
                if rec_gene.inno_num < child_genes.inno_num {
                    rec_iter.next();
                }
                else {
                    break;
                }
            }

            if let Some(&rec_gene) = rec_iter.peek(){
                if rec_gene.inno_num == child_genes.inno_num {
                    if rng.random_bool(0.5) {
                        child_genes.weight = rec_gene.weight;
                    }
                    if !child_genes.enabled || !rec_gene.enabled {
                        child_genes.enabled = !rng.random_bool(INHERIT_DISABLE_PROB);
                    }

                    rec_iter.next();
                }
            }
        }

        (child_neurons, child_synapses, dom_species)
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
                        let new_num = self.new_inno_index;
                        self.new_inno_index += 1;
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
            let target_idx = rng.random_range(0..synapse_genes.len());
            let old_synapse = synapse_genes[target_idx].clone();
            synapse_genes[target_idx].enabled = false;

            let new_node_id = *new_nodes.entry(old_synapse.inno_num).or_insert_with(|| {
                let id = self.new_node_index;
                self.new_node_index += 1;
                id
            });

            let is_new_split = new_node_id == self.new_node_index - 1;

            let (inno1, inno2) = if is_new_split {
                let i1 = self.new_inno_index;
                let i2 = self.new_inno_index + 1;
                self.new_inno_index += 2;

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
                    break 'found_index Some((active_index, anchor.species_hist_index()))
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
                let species_hist_index = self.new_species_index;

                new_indices.push(Vec::with_capacity(1));
                self.active_species.push(ActiveSpecies::new(species_hist_index));
                self.history.species.push(SpeciesHistory{
                    id: species_hist_index,
                    birth_gen: self.generation_index,
                    origin_species,
                    extinct_gen: None,
                    max_fitness: Vec::new(),
                    generation_indices: Vec::new(),
                });

                (active_index, species_hist_index, true)
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

