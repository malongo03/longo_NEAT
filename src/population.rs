use crate::genome::{genome_crossover, genome_distance, Genome, NeuronGene, SynapseGene};
use crate::mutation::{mutate, mutate_no_structure_change, MutationParameters};
use rand::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use crate::history::{History, SpeciesHistory};

#[derive(Debug, Clone)]
pub struct Generation(Vec<Genome>);
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

// Eventually, these will all be made actual settings
const INHERIT_DISABLE_PROB: f64 = 0.75;

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

    gens_since_best_fitness: usize, // Generations from last improvement. Default is 0.
    best_fitness: f64, // Last significant best fitness
}
impl ActiveSpecies {
    fn new(species_history_id: usize) -> Self {
        Self {
            species_history_id,
            member_indices: Vec::new(),
            gens_since_best_fitness: 0,
            best_fitness: f64::NEG_INFINITY,
        }
    }
}

pub struct Population {
    population_size: NonZeroUsize,
    species_threshold: f64,
    mutation_parameters: MutationParameters,

    active_species: Vec<ActiveSpecies>,
    curr_gen: Generation,
    fitness: Vec<Option<f64>>,

    next_inno_index: usize,
    next_node_index: usize,
    next_species_id: usize,
    generation_index: usize,

    stagnations_since_best_fitness: usize,
    gens_since_best_fitness: usize, // Generations from last improvement. Default is 0.
    best_fitness: f64, // Best fitness so far
    history: History
}
impl Population {

    // TODO: Better error handling with a proper error type
    // TODO: Some of this functionality can definitely be merged with next_generation
    pub fn new_from_base<R: Rng>(rng: &mut R,
                                 population_size: NonZeroUsize,
                                 species_threshold: f64,
                                 starting_genome: Genome)
        -> Result<Self, Box<dyn Error>> {

        if species_threshold <= 0.0 {
            return Err("species_threshold must be strictly positive".into());
        }
        starting_genome.check_assumptions()?;

        let mutation_parameters = MutationParameters::default_params();

        let fitness: Vec<Option<f64>> = vec![None; population_size.get()];

        let next_inno_index: Option<&SynapseGene> = starting_genome.synapse_genes().last();
        let next_inno_index: usize = if let Some(gene) = next_inno_index {gene.inno_num() + 1} else {0};
        let next_node_index: Option<&NeuronGene> = starting_genome.neuron_genes().last();
        let next_node_index: usize = if let Some(gene) = next_node_index {gene.node_name() + 1} else {0};
        let mut next_species_id: usize = 1;
        let generation_index: usize = 0;

        let stagnations_since_best_fitness: usize = 0;
        let gens_since_best_fitness = 0;
        let best_fitness: f64 = f64::NEG_INFINITY;

        // Populating population //

        let mut active_species: Vec<ActiveSpecies> = vec![ActiveSpecies::new(0)];
        let mut population: Generation = Generation(Vec::with_capacity(population_size.get()));
        let mut anchors: Vec<Genome> = Vec::new();
        let mut history = History::new_blank();
        history.species.push(SpeciesHistory::new(0, 0, None));

        // Create clone of the starting genome to act as an initial anchor
        let seed_genome = Genome::new_unchecked(
            0,
            0,
            starting_genome.neuron_genes().clone(),
            starting_genome.synapse_genes().clone(),
        );
        anchors.push(seed_genome.clone());
        population.push(seed_genome);
        active_species[0].member_indices.push(0);

        for id in 1..population_size.get() {
            let mutated_synapses = mutate_no_structure_change(rng, &mutation_parameters, starting_genome.synapse_genes().clone());

            let species_index: Option<usize> = anchors
                .iter()
                .enumerate()
                .find(|(_, anchor)| {
                    genome_distance(
                        &mutated_synapses,
                        anchor.synapse_genes(),
                        DISTANCE_DISJOINT_WEIGHT,
                        DISTANCE_EXCESS_WEIGHT,
                        DISTANCE_WEIGHT_DIFF_WEIGHT,
                    ) <= species_threshold
                })
                .map(|(active_index, _)| active_index);

            let (species_index, push_anchor) =
                if let Some(index) = species_index {
                    (index, false)
                }
                else {
                    let index = next_species_id;
                    next_species_id += 1;

                    active_species.push(ActiveSpecies::new(index));
                    history.species.push(SpeciesHistory::new(index, 0, Some(0)));

                    (index, true)
                };

            active_species[species_index].member_indices.push(id);

            let genome = Genome::new_unchecked(id, species_index, starting_genome.neuron_genes().clone(), mutated_synapses);

            if push_anchor {anchors.push(genome.clone())}

            population.push(genome);
        }

        Ok(Self{
            population_size,
            species_threshold,
            mutation_parameters,

            active_species,
            curr_gen: population,
            fitness,

            next_inno_index,
            next_node_index,
            next_species_id,
            generation_index,

            stagnations_since_best_fitness,
            gens_since_best_fitness,
            best_fitness,

            history,
        })
    }

    pub fn insert_fitness(&mut self,
                          fitness: f64,
                          id: usize) -> Result<(), Box<dyn Error>> {
        if fitness.is_nan() {
            return Err("Inserted fitness should not be NaN".into())
        }
        if fitness.is_infinite() {
            return Err("Inserted fitness should not be infinite".into())
        }
        if id >= self.population_size.get() {
            return Err("Population id out of bounds".into())
        }
        self.fitness[id] = Some(fitness);

        Ok(())
    }

    fn replace_fitness_vector_unchecked(&mut self, new_fitness_vector: &[f64]) {
        for (idx, &fitness) in new_fitness_vector.iter().enumerate() {
            self.fitness[idx] = Some(fitness);
        }
    }

    pub fn new_generation<R: Rng>(&mut self, rng: &mut R) {

        self.generation_index += 1;

        // 1. Perform fitness calculations and cull species population
        let (best_fitness_per_species,
            average_fitness_per_species,
            champions,
            anchors) = self.calculate_species_fitness_and_cull(rng);

        // 2. Check which species are stagnating
        let stagnating_species: Vec<bool> = self.check_for_stagnation(best_fitness_per_species);

        // 3. Determine allotments for each active species
        let allotted_children = Population::allot_children(self.population_size.get(), average_fitness_per_species, stagnating_species);

        // 4. Produce new generation, sorted into proper species
        let (new_indices, new_generation) = self.reproduce(rng, champions, anchors, allotted_children);

        // 5. Perform extinctions
        let new_active_species = self.perform_extinctions(new_indices);

        // 6. Move everything
        self.curr_gen = new_generation;
        self.active_species = new_active_species;
        for fitness in self.fitness.iter_mut() {
            *fitness = None;
        }
    }

    fn calculate_species_fitness_and_cull<R: Rng>(&mut self,
                                                  rng: &mut R)
        -> (Vec<f64>, Vec<f64>, Generation, Generation) {
        // 0.) Sanitize fitness scores. We default to a fitness of 0.0 if no fitness has been assigned.
        for fitness in self.fitness.iter_mut() {
            fitness.get_or_insert(0.0);
        }

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
            champions.push(self.curr_gen[champion_index].clone());
            best_fitness_per_species.push(champion_fitness);

            // 2
            let repr_index: usize = *active_species.member_indices.choose(rng).expect(
                "An empty active species should have been made extinct and filtered out."
            );
            anchors.push(self.curr_gen[repr_index].clone());

            // 3
            let species_fitness: f64 = active_species.member_indices.iter().map(|idx| {self.fitness[*idx].expect("Already sanitized.")}).sum();
            let species_pop: f64 = active_species.member_indices.len() as f64;
            average_fitness_per_species.push(species_fitness / species_pop);

            // 4
            let survival_count = active_species.member_indices.len() - (EVOLUTION_TRUNCATION * species_pop).floor() as usize;
            active_species.member_indices.truncate(survival_count);

            // 5
            self.history.species[active_species.species_history_id].max_fitness.push(champion_fitness);
            self.history.species[active_species.species_history_id].history_index_by_generation.push(active_index);
        }

        // 6. Save current generation's genomes for historical analysis
        self.history.reprs.push(anchors.clone());
        self.history.bests.push(champions.clone());

        (best_fitness_per_species, average_fitness_per_species, champions, anchors)
    }

    fn check_for_stagnation(&mut self,
                            best_fitness_per_species: Vec<f64>)
        -> Vec<bool> {
        let num_active_species = self.active_species.len();

        let mut stagnating_species: Vec<bool> = vec![false; num_active_species];
        if num_active_species == 1 { // No population or species stagnation can occur during zero diversity

            let top1_score: f64 = best_fitness_per_species[0];

            self.history.max_fitness.push(top1_score);

            self.stagnations_since_best_fitness = 0;
            self.gens_since_best_fitness = 0;
            self.best_fitness = top1_score;

            let active_species = &mut self.active_species[0];

            active_species.gens_since_best_fitness = 0;
            active_species.best_fitness = top1_score;
            return stagnating_species
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
        // will be allotted zero children.
        let population_stagnation: bool = {
            self.gens_since_best_fitness += 1;
            if top1_score - self.best_fitness > EVOLUTION_STAGNATION_EPSILON {
                self.stagnations_since_best_fitness += 1;
                self.gens_since_best_fitness = 0;
                self.best_fitness = top1_score;
                false
            }
            else if self.gens_since_best_fitness >= 20 {
                self.stagnations_since_best_fitness += 1;
                self.gens_since_best_fitness = 0;
                self.best_fitness = top1_score;
                true
            }
            else {
                false
            }
        };
        if population_stagnation {
            for (index, stagnation) in stagnating_species.iter_mut().enumerate() {
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
            active_species.gens_since_best_fitness += 1;
            if fitness - active_species.best_fitness > EVOLUTION_STAGNATION_EPSILON {
                active_species.gens_since_best_fitness = 0;
                active_species.best_fitness = fitness;
                continue;
            }
            // Elitism protection. Top two species can never stagnate.
            if active_index == top1_index || active_index == top2_index {
                continue;
            }
            if active_species.gens_since_best_fitness >= 15 {
                stagnating_species[active_index] = true;
            }
        }

        stagnating_species
    }

    fn allot_children(population_size: usize,
                      average_fitness_per_species: Vec<f64>,
                      stagnating_species: Vec<bool>)
        -> Vec<usize> {

        let num_active_species: usize = average_fitness_per_species.len();

        // Determine total fitness of non-stagnating species
        let total_fitness: f64 = average_fitness_per_species.iter()
            .enumerate()
            .filter(|(idx, _)| !stagnating_species[*idx])
            .map(|(_, &fitness)| fitness)
            .sum();

        let mut remainder: usize = population_size;

        // Assign respective fraction of the new generation: avg_fitness / total_fitness rounded down
        let mut allotted_children: Vec<usize> = vec![0; num_active_species];
        for (active_index, avg_fitness) in average_fitness_per_species.into_iter().enumerate() {
            if stagnating_species[active_index] {
                continue;
            }
            let allotment: usize = (avg_fitness / total_fitness).floor() as usize;
            allotted_children[active_index] = allotment;
            remainder -= allotment;
        }

        // Round-robin assign any remaining allotment to the youngest species.
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

        allotted_children
    }

    fn reproduce<R: Rng>(&mut self,
                         mut rng: &mut R,
                         champions: Generation,
                         mut anchors: Generation,
                         allotted_children: Vec<usize>)
                         -> (Vec<Vec<usize>>, Generation) {

        let num_active_species: usize = champions.len();

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
                let (neuron_genes, synapse_genes, origin_species) = self.breed_species(rng, num_active_species, active_index, num_potential_parents);

                let (neuron_genes, synapse_genes) = mutate(&mut rng,
                                                           &self.mutation_parameters,
                                                           &mut self.next_inno_index,
                                                           &mut self.next_node_index,
                                                           &mut new_innos,
                                                           &mut new_nodes,
                                                           neuron_genes,
                                                           synapse_genes);

                self.sort_and_place_new_genome(&mut new_generation,
                                               &mut new_indices,
                                               &mut anchors,
                                               neuron_genes,
                                               synapse_genes,
                                               origin_species);
            }
        }
        (new_indices, new_generation)
    }

    #[inline]
    fn breed_species<R: Rng>(&mut self,
                             mut rng: &mut R,
                             num_active_species: usize,
                             active_index: usize,
                             num_potential_parents: usize)
        -> (Vec<NeuronGene>, Vec<SynapseGene>, usize) {
        let parent1_index = *self.active_species[active_index].member_indices
            .choose(&mut rng).expect("Pop indices should always be non-empty");
        let parent1: &Genome = &self.curr_gen[parent1_index];
        let fitness1: f64 = self.fitness[parent1_index].expect(
            "The fitness vector should already have been sanitized."
        );

        if rng.random_bool(EVOLUTION_NO_CROSSOVER) {
            let genome = &self.curr_gen[parent1_index];
            (genome.neuron_genes().clone(), genome.synapse_genes().clone(), genome.species_history_id())
        }

        // Interspecies Crossover
        else if num_active_species > 1 && rng.random_bool(EVOLUTION_INTERSPECIES) {
            let species_index = rng.random_range(0..num_active_species - 1);
            let species_index =
                if active_index <= species_index {
                    species_index + 1
                } else {
                    species_index
                };
            let parent2_index = *self.active_species[species_index].member_indices
                .choose(&mut rng).expect("Pop indices will always be non-empty");
            let parent2: &Genome = &self.curr_gen[parent2_index];
            let fitness2: f64 = self.fitness[parent2_index].expect(
                "The fitness vector should already have been sanitized."
            );

            genome_crossover(parent1, parent2, fitness1, fitness2, INHERIT_DISABLE_PROB)
        }

        // Normal Crossover
        else {
            if num_potential_parents > 1 {
                // Search for a valid partner for parent 1
                for _ in 0..20 {
                    let parent2_index = *self.active_species[active_index].member_indices
                        .choose(&mut rng).expect("Pop indices will always be non-empty");
                    if parent1_index == parent2_index {
                        continue;
                    }
                    let parent2: &Genome = &self.curr_gen[parent2_index];
                    let fitness2: f64 = self.fitness[parent2_index].expect(
                        "The fitness vector should already have been sanitized."
                    );

                    return genome_crossover(parent1, parent2, fitness1, fitness2, INHERIT_DISABLE_PROB);
                }
            }
            // Default path: just clone parent 1
            (parent1.neuron_genes().clone(), parent1.synapse_genes().clone(), parent1.species_history_id())
        }
    }

    fn sort_and_place_new_genome(&mut self,
                                 new_generation: &mut Generation,
                                 new_indices: &mut Vec<Vec<usize>>,
                                 anchors: &mut Generation,
                                 new_neuron_genes: Vec<NeuronGene>,
                                 new_synapse_genes: Vec<SynapseGene>,
                                 origin_species: usize) {
        let id = new_generation.len();

        let species_indices: Option<(usize, usize)> = anchors
            .iter()
            .enumerate()
            .find(|(_, anchor)| {
                genome_distance(
                    &new_synapse_genes,
                    anchor.synapse_genes(),
                    DISTANCE_DISJOINT_WEIGHT,
                    DISTANCE_EXCESS_WEIGHT,
                    DISTANCE_WEIGHT_DIFF_WEIGHT,
                ) <= self.species_threshold
            })
            .map(|(active_index, anchor)| (active_index, anchor.species_history_id()));

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
                self.history.species.push(SpeciesHistory::new(species_history_id, self.generation_index, Some(origin_species)));

                (active_index, species_history_id, true)
            };

        new_indices[active_index].push(id);

        let genome = Genome::new(id,
                                 hist_index,
                                 new_neuron_genes,
                                 new_synapse_genes)
            .expect("new_generation should not produce an invalid Genome");

        if push_anchor {anchors.push(genome.clone());}

        new_generation.push(genome);
    }

    fn perform_extinctions(&mut self,
                           new_indices: Vec<Vec<usize>>)
        -> Vec<ActiveSpecies> {
        let num_active_species: usize = self.active_species.len();

        let mut new_active_species: Vec<ActiveSpecies> = Vec::with_capacity(num_active_species);
        for (active_index, member_indices) in new_indices.into_iter().enumerate() {
            let old_species = &self.active_species[active_index];
            if member_indices.len() == 0 {
                self.history.species[old_species.species_history_id].extinct_gen = Some(self.generation_index);
                continue;
            }
            new_active_species.push(ActiveSpecies {
                species_history_id: old_species.species_history_id,
                member_indices,
                gens_since_best_fitness: old_species.gens_since_best_fitness,
                best_fitness: old_species.best_fitness,
            });
        }
        new_active_species
    }

}