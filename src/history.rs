use crate::population::Generation;

// This is split into its own 

pub struct SpeciesHistory {
    pub id: usize,
    pub birth_gen: usize,
    pub origin_species: Option<usize>,
    pub extinct_gen: Option<usize>,
    pub max_fitness: Vec<f64>,
    pub history_index_by_generation: Vec<usize>,
}

impl SpeciesHistory {
    pub fn new(id: usize, birth_gen: usize, origin_species: Option<usize>) -> Self {
        Self {
            id,
            birth_gen,
            origin_species,
            extinct_gen: None,
            max_fitness: vec![],
            history_index_by_generation: vec![],
        }
    }
}

pub struct History {
    pub reprs: Vec<Generation>,
    pub bests: Vec<Generation>,
    pub max_fitness: Vec<f64>,
    pub species: Vec<SpeciesHistory>,
}

impl History {
    pub fn new_blank() -> Self {
        Self {
            reprs: vec![],
            bests: vec![],
            max_fitness: vec![],
            species: vec![],
        }
    }
}