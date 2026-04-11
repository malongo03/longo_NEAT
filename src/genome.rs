use std::cmp::max;
use std::collections::HashSet;
use std::error::Error;
use std::fmt::Debug;
use rand::prelude::*;

/// The type of a neuron described by a NetworkNode or a NeuronGene.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuronType {
    /// A neuron that takes in only a simulation/sensory input.
    /// Its internal value points to its input source index.
    Sensory(usize),
    /// A neuron whose inputs and outputs are only other neurons.
    Inter(),
    /// A neuron that gives output from the network for the simulation.
    /// Muscular neurons are also permitted to output to other neurons.
    /// Its internal value points to its simulation output index.
    Muscular(usize)
}

/// The gene of one neuron within a Genome's described neural network.
#[derive(Debug, Clone)]
pub struct NeuronGene {
    /// Global name of this node (used for cross-over homology)
    pub node_name: usize,
    pub neuron_type: NeuronType
}

/// The gene of a connection between two neurons in a Genome.
#[derive(Debug, Clone)]
pub struct SynapseGene {
    /// Index of neuron sending signal.
    pub src_id: usize,
    /// Index of neuron receiving signal.
    pub tgt_id: usize,
    /// Weight multiplier of signal.
    pub weight: f64,
    /// Innovation number of this gene. (See: mutation,
    pub inno_num: usize,
    /// If this edge appears in the phenotype.
    pub enabled: bool
}

/// A description of a neural network that can be used for evolution. Genomes cannot be edited once
/// created.
///
/// If you wish to create a custom Genome, you must either create a new Genome from
/// a list of genes, or create a Neural_Network object and convert it to a Genome. (Be aware that
/// this latter option will erase any evolution data in the nodes and synapses).
///
/// # Members (accessed only through getters)
///
/// ***id: usize*** - Index of a genome within a Population.
///
/// ***species_history_id: usize*** - Index of the species within the History of a Population.
///
/// ***neuron_genes: Vec<NeuronGene>*** - List of genes for nodes in the neural network.
///
/// ***synapse_genes: Vec<SynapseGene>*** - List of genes for edges in the neural network.
///
/// # Interface Contract
///
/// *neuron_genes* is sorted by node_name and has no duplicates
///
/// *neuron_genes* contains at least one Sensory NeuronGene and one Muscular NeuronGene
///
/// *synapse_genes* is sorted by inno_num and has no duplicates
///
/// *synapse_genes* has no duplicate edges
///
/// *synapse_genes* contains only the names of nodes that exist within *node_genes*
///
/// Sensory type neuron have no input edges (that is, their only input is their input source)
///
/// Inter type neuron have at least one input edge and one output edge (though the genes may be disabled)
///
/// Muscular type neuron have at least one input edge (though the gene may be disabled)
#[derive(Debug, Clone)]
pub struct Genome {
    /// Index of a genome within a Population.
    id: usize,
    /// Index of the species as it arose in evolution.
    species_history_id: usize,
    neuron_genes: Vec<NeuronGene>,
    synapse_genes: Vec<SynapseGene>,
}
impl Genome {
    /// Create a Genome object. This function guarantees the Genome is compatible with Population's
    /// and Neural_Network's assumptions.
    pub fn new(id: usize, 
               species_history_id: usize, 
               neuron_genes: Vec<NeuronGene>,
               synapse_genes: Vec<SynapseGene>) 
        -> Result<Genome, Box<dyn Error>> {

        let new_genome = Self {
            id,
            species_history_id,
            neuron_genes,
            synapse_genes
        };

        if let Err(error) = new_genome.check_assumptions(){
            return Err(error);
        }
        Ok(new_genome)
    }

    /// Create a Genome object without a guarantee that it is valid for Population,
    /// Neural_Network, and genome_distance. This should only ever be used if your function has been
    /// validated to never trigger a check_assumptions error.
    pub fn new_no_check(id: usize, 
                        species_history_id: usize, 
                        neuron_genes: Vec<NeuronGene>,
                        synapse_genes: Vec<SynapseGene>) 
        -> Genome {
        Self{id, species_history_id, neuron_genes, synapse_genes}
    }

    /// Checks for the following possible violations of Genome's parameters:
    ///
    /// *neuron_genes* has duplicate node_names or is not sorted by node_name.
    ///
    /// *synapse_genes* has duplicate inno_nums or is not sorted by inno_num.
    ///
    /// *synapse_genes* has duplicate edges.
    ///
    /// *synapse_genes* contains the names of nodes that do not exist in *node_genes*
    ///
    /// Sensory type neuron has an input edge
    ///
    /// Inter type neuron has no input edges or no output edges
    ///
    /// Muscular type neuron has no input edges
    ///
    /// The Genome has no Sensory neurons.
    ///
    /// The Genome has no Muscular neurons.
    pub fn check_assumptions(&self) 
        -> Result<(), Box<dyn Error>> {
        let mut seen_nodes: HashSet<usize> = HashSet::with_capacity(self.neuron_genes.len());
        let mut prev_name: Option<usize> = None;

        for node in &self.neuron_genes {
            if prev_name.is_some_and(|prev| prev >= node.node_name) {
                return Err("Genome nodes names should be strictly ascending.".into())
            }
            prev_name = Some(node.node_name);

            seen_nodes.insert(node.node_name);
        }

        let mut prev_inno: Option<usize> = None;
        let mut seen_edges: HashSet<(usize, usize)> = HashSet::with_capacity(self.synapse_genes.len());
        let mut is_src: HashSet<usize> = HashSet::with_capacity(self.neuron_genes.len());
        let mut is_tgt: HashSet<usize> = HashSet::with_capacity(self.neuron_genes.len());

        for gene in &self.synapse_genes {
            if prev_inno.is_some_and(|prev| prev >= gene.inno_num) {
                return Err("Genome innovation numbers should be strictly ascending.".into())
            }
            prev_inno = Some(gene.inno_num);

            let edge_pair: (usize, usize) = (gene.src_id, gene.tgt_id);
            if !seen_edges.insert(edge_pair) {
                return Err("Genome contains duplicate edge pair.".into())
            }
            if !seen_nodes.contains(&gene.src_id) || !seen_nodes.contains(&gene.tgt_id) {
                return Err("synapse_genes should not have edges to or from nodes the Genome lacks.".into())
            }

            is_src.insert(gene.src_id);
            is_tgt.insert(gene.tgt_id);
        }

        let mut has_sensory: bool = false;
        let mut has_muscular: bool = false;

        for node in &self.neuron_genes {
            match node.neuron_type {
                NeuronType::Sensory(_) => {
                    has_sensory = true;
                    if is_tgt.contains(&node.node_name) {
                        return Err("Sensory neurons should have no input edges.".into());
                    }
                }
                NeuronType::Inter() => {
                    if !is_tgt.contains(&node.node_name) {
                        return Err("Inter neurons should have at least one input edge.".into());
                    }
                    if !is_src.contains(&node.node_name) {
                        return Err("Inter neurons should have at least one output edge.".into());
                    }
                }
                NeuronType::Muscular(_) => {
                    has_muscular = true;
                    if !is_tgt.contains(&node.node_name) {
                        return Err("Muscular neurons should have at least one input edge.".into());
                    }
                }
            }
        }

        if !has_sensory {
            return Err("A Genome should have at least one Sensory neuron.".into());
        }
        if !has_muscular {
            return Err("A Genome should have at least one Muscular neuron.".into());
        }

        Ok(())
    }

    // GETTERS

    /// In NEAT, genome size is the number of edge/synapse genes (not the number of nodes).
    #[inline]
    pub fn genome_size(&self) -> usize {self.synapse_genes.len()}

    // Index of a genome within a Population.
    #[inline]
    pub fn id(&self) -> usize {self.id}

    /// Index of the species within the History of a Population.
    #[inline]
    pub fn species_history_id(&self) -> usize {self.species_history_id }

    /// List of genes for nodes in the neural network.
    #[inline]
    pub fn neuron_genes(&self) -> &Vec<NeuronGene> {&self.neuron_genes}

    /// List of genes for edges in the neural network.
    #[inline]
    pub fn synapse_genes(&self) -> &Vec<SynapseGene> {&self.synapse_genes}
}

/// The genetic distance of two genome, weighted by some constants.
///
/// Genes are matched up homology (that is, whether they share the same innovation id). Some genes
/// will only exist in one genome. If the innovation number of this unique gene is
/// below the maximum innovation number of the other genome, it is considered a "disjoint" gene. If
/// it is beyond the maximum innovation number of the other genome, it is considered an
/// "excess" gene.
///
/// Another way to think about: disjoint genes are ancient genes that one genome lacks, while
/// excess genes are modern genes that one genome lacks.
///
/// # Parameters
///
/// **first, second: &Genome** - The two genomes to be compared.
///
/// **disjoint_weight: f64** - The weighting of "disjoint" genes.
///
/// **excess_weight: f64** - The weighting of "excess" genes.
///
/// **weight_diff_weight: f64** - The weighting of the average weight difference between shared genes
pub fn genome_distance(first: &Vec<SynapseGene>, second: &Vec<SynapseGene>,
                       disjoint_weight: f64, excess_weight: f64, weight_diff_weight: f64) -> f64 {
    let mut iter1 = first.iter().peekable();
    let mut iter2 = second.iter().peekable();

    let size_factor: f64 = max(first.len(), second.len()) as f64;
    let size_factor: f64 = if size_factor < 20.0 { 1.0 } else { size_factor };

    let mut excess: f64 = 0.0;
    let mut disjoint: f64 = 0.0;
    let mut shared: f64 = 0.0;

    let mut weight_diff: f64 = 0.0;

    while let (Some(gene1), Some(gene2)) = (iter1.peek(), iter2.peek()) {
        if gene1.inno_num == gene2.inno_num {
            shared += 1.0;
            weight_diff += (gene1.weight - gene2.weight).abs();

            iter1.next();
            iter2.next();
        }
        else {
            disjoint += 1.0;
            if gene1.inno_num < gene2.inno_num {
                iter1.next();
            }
            else {
                iter2.next();
            }
        }
    }

    for _ in iter1 {
        excess += 1.0;
    }
    for _ in iter2 {
        excess += 1.0;
    }

    let mismatch_penalty: f64 = (excess_weight * excess + disjoint_weight * disjoint) / size_factor;
    let weight_penalty: f64 = if shared > 0.0 { (weight_diff_weight * weight_diff) / shared } else { 0.0 };

    mismatch_penalty + weight_penalty
}

pub fn genome_crossover(genome1: &Genome,
                        genome2: &Genome,
                        fitness1: f64,
                        fitness2: f64,
                        inherit_disable_prob: f64) -> (Vec<NeuronGene>, Vec<SynapseGene>, usize) {

    // Genome with the highest fitness is considered dominant. If neither has a higher fitness,
    // we consider both Genomes dominant.
    let (dom_neur, dom_syna, dom_species, rec_neur, rec_syna, both_dom) =
        if fitness1 > fitness2 {
            (genome1.neuron_genes(),
             genome1.synapse_genes(),
             genome1.species_history_id(),
             genome2.neuron_genes(),
             genome2.synapse_genes(),
             false)
        }
        else if fitness2 > fitness1 {
            (genome2.neuron_genes(),
             genome2.synapse_genes(),
             genome2.species_history_id(),
             genome1.neuron_genes(),
             genome1.synapse_genes(),
             false)
        }
        else {
            (genome1.neuron_genes(),
             genome1.synapse_genes(),
             genome1.species_history_id(),
             genome2.neuron_genes(),
             genome2.synapse_genes(),
             true)
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
        while let Some(&node) = neur_iter1.peek() {
            new_neurons.push(node.clone());
            neur_iter1.next();
        }
        while let Some(&node) = neur_iter2.peek() {
            new_neurons.push(node.clone());
            neur_iter2.next();
        }

        let mut syna_iter1 = genome1.synapse_genes().iter().peekable();
        let mut syna_iter2 = genome2.synapse_genes().iter().peekable();

        while let (Some(&gene1), Some(&gene2)) = (syna_iter1.peek(), syna_iter2.peek()) {
            if gene1.inno_num < gene2.inno_num {
                new_synapses.push(gene1.clone());
                syna_iter1.next();
            }
            else if gene2.inno_num < gene1.inno_num {
                new_synapses.push(gene2.clone());
                syna_iter2.next();
            }
            else {
                let mut syna_gene = if rng.random_bool(0.5) {
                    gene1.clone()
                }
                else {
                    gene2.clone()
                };

                if !gene1.enabled || !gene2.enabled {
                    syna_gene.enabled = !rng.random_bool(inherit_disable_prob);
                }

                new_synapses.push(syna_gene);
                syna_iter1.next();
                syna_iter2.next();
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
                    child_genes.enabled = !rng.random_bool(inherit_disable_prob);
                }

                rec_iter.next();
            }
        }
    }

    (child_neurons, child_synapses, dom_species)
}


#[cfg(test)]
mod genome_tests {
    use super::*;
    use test_case::test_case;

    fn helper_mock_synapse(inno_num: usize, weight: f64) -> SynapseGene {
        SynapseGene{
            src_id: 0,
            tgt_id: 1,
            weight,
            inno_num,
            enabled: true,
        }
    }

    fn helper_mock_genome(synapses: Vec<SynapseGene>) -> Genome {
        Genome::new_no_check(0, 0, vec![], synapses)
    }
    
    fn helper_mock_valid_network() -> (Vec<NeuronGene>, Vec<SynapseGene>) {
        let neurons = vec![
            NeuronGene{node_name: 0, neuron_type: NeuronType::Sensory(0)},
            NeuronGene{node_name: 1, neuron_type: NeuronType::Inter()},
            NeuronGene{node_name: 2, neuron_type: NeuronType::Muscular(0)}
        ];
        let synapses = vec![
            SynapseGene{ src_id:0, tgt_id: 1, weight: 1.0, inno_num: 0, enabled: true},
            SynapseGene{ src_id:1, tgt_id: 2, weight: 1.0, inno_num: 1, enabled: true},
        ];

        (neurons, synapses)
    }
    
    /////////////////////////////////
    // VALID/INVALID GENOME TESTS //
    ////////////////////////////////

    #[test]
    fn test_valid_base_network() {
        let (n, s) = helper_mock_valid_network();
        assert!(Genome::new(5, 5, n, s).is_ok(),
                "The basic test network should be a valid genome")
    }

    #[test]
    fn test_invalid_unsorted_neurons() {
        let (mut n, s) = helper_mock_valid_network();
        n.swap(0, 1);
        assert!(Genome::new(5, 5, n, s).is_err(),
                "Genome::new should return an error if neuron_genes is not sorted by node_name");
    }

    #[test]
    fn test_invalid_duplicate_node_names() {
        let (mut n, s) = helper_mock_valid_network();
        n[1].node_name = 0; // Two node_name 0s
        assert!(Genome::new(5, 5, n, s).is_err(),
                "Genome::new should return an error if there are duplicate node names");
    }

    #[test]
    fn test_invalid_unsorted_synapses() {
        let (n, mut s) = helper_mock_valid_network();
        s.swap(0, 1);
        assert!(Genome::new(5, 5, n, s).is_err(),
                "Genome::new should return an error if synapses are not sorted by inno_num");
    }

    #[test]
    fn test_invalid_duplicate_inno_nums() {
        let (n, mut s) = helper_mock_valid_network();
        s[1].inno_num = 0;
        assert!(Genome::new(5, 5, n, s).is_err(),
                "Genome::new should return an error if there are duplicate inno nums");
    }

    #[test]
    fn test_invalid_duplicate_edge() {
        let (n, mut s) = helper_mock_valid_network();
        s[1].src_id = 0;
        s[1].tgt_id = 1;
        assert!(Genome::new(5, 5, n, s).is_err(),
                "Genome:new should return an error if there is a duplicate edge");
    }

    #[test]
    fn test_invalid_missing_node_reference() {
        let (n, mut s) = helper_mock_valid_network();
        s[0].tgt_id = 99;
        assert!(Genome::new(5, 5, n, s).is_err(),
                "Genome:new should return an error if node is referenced in synapse_genes that is \
                missing in node_genes");
    }

    #[test]
    fn test_invalid_sensory_with_input_edge() {
        let (n, mut s) = helper_mock_valid_network();
        s.push(SynapseGene { src_id: 2, tgt_id: 0, weight: 1.0, inno_num: 2, enabled: true });
        assert!(Genome::new(5, 5, n, s).is_err(),
                "Genome::new should return an error if a Sensory neuron has an input edge");
    }

    #[test]
    fn test_invalid_inter_missing_input_edge() {
        let (n, mut s) = helper_mock_valid_network();
        s.remove(0);
        assert!(Genome::new(5, 5, n, s).is_err(),
                "Genome::new should return an error if an Inter neuron has no input edge");
    }

    #[test]
    fn test_invalid_inter_missing_output_edge() {
        let (n, mut s) = helper_mock_valid_network();
        s.remove(1);
        assert!(Genome::new(5, 5, n, s).is_err(),
                "Genome::new should return an error if an Inter neuron has no output edge");
    }

    #[test]
    fn test_invalid_muscular_missing_input_edge() {
        let (n, mut s) = helper_mock_valid_network();
        s[1].tgt_id = 1;
        assert!(Genome::new(5, 5, n, s).is_err(),
                "Genome::new should return an error if a muscular neuron has no input edge");
    }

    #[test]
    fn test_invalid_missing_sensory_neuron() {
        let no_sensory_n = vec![
            NeuronGene { node_name: 1, neuron_type: NeuronType::Inter() },
            NeuronGene { node_name: 2, neuron_type: NeuronType::Muscular(0) },
        ];
        let no_sensory_s = vec![
            SynapseGene { src_id: 1, tgt_id: 1, weight: 1.0, inno_num: 0, enabled: true },
            SynapseGene { src_id: 1, tgt_id: 2, weight: 1.0, inno_num: 1, enabled: true },
        ];
        assert!(Genome::new(5, 5, no_sensory_n, no_sensory_s).is_err(),
                "Genome::new should return an error if there is no Sensory neuron in neuron_genes");
    }

    #[test]
    fn test_invalid_missing_muscular_neuron() {
        let no_muscular_n = vec![
            NeuronGene { node_name: 0, neuron_type: NeuronType::Sensory(0) },
            NeuronGene { node_name: 1, neuron_type: NeuronType::Inter() },
        ];
        let no_muscular_s = vec![
            SynapseGene { src_id: 0, tgt_id: 1, weight: 1.0, inno_num: 0, enabled: true },
            SynapseGene { src_id: 1, tgt_id: 1, weight: 1.0, inno_num: 1, enabled: true },
        ];
        assert!(Genome::new(5, 5, no_muscular_n, no_muscular_s).is_err(),
                "Genome::new should return an error if there is no Muscular neuron in neuron_genes");
    }

    ////////////////////////////////
    // GENOME DISTANCE MATH TESTS //
    ////////////////////////////////

    // Format: #[test_case(synapses1, synapses2, d_weight, e_weight, w_weight, expected_distance)]
    #[test_case(
        vec![helper_mock_synapse(1, 1.0), helper_mock_synapse(2, 1.0)],
        vec![helper_mock_synapse(1, 1.0), helper_mock_synapse(2, 1.0)],
        1.0, 1.0, 1.0, 0.0
        ; "Identical genomes (Distance = 0)"
    )]
    #[test_case(
        vec![helper_mock_synapse(1, 1.0)],
        vec![helper_mock_synapse(1, 2.0)],
        1.0, 1.0, 1.0, 1.0
        ; "Weight difference only"
    )]
    #[test_case(
        vec![helper_mock_synapse(1, 1.0), helper_mock_synapse(2, 1.0)],
        vec![helper_mock_synapse(1, 1.0), helper_mock_synapse(3, 1.0), helper_mock_synapse(4, 1.0)],
        3.0, 2.0, 1.0, 7.0
        ; "Disjoint and Excess combinations"
    )]
    #[test_case(
        vec![helper_mock_synapse(1, 1.0), helper_mock_synapse(2, 1.0)],
        vec![helper_mock_synapse(3, 1.0), helper_mock_synapse(4, 1.0)],
        1.0, 1.0, 100.0, 4.0
        ; "Completely disjoint (No shared genes / Safe div by zero)"
    )]
    #[test_case(
        (1..=20).map(|i| helper_mock_synapse(i, 1.0)).collect(),
        { let mut s: Vec<SynapseGene> = (1..=19).map(|i| helper_mock_synapse(i, 1.0)).collect(); s.push(helper_mock_synapse(21, 1.0)); s },
        1.0, 1.0, 1.0, 0.1
        ; "Size factor scaling (> 20 genes)"
    )]
    fn test_genome_distance_math(
        s1: Vec<SynapseGene>,
        s2: Vec<SynapseGene>,
        d_weight: f64,
        e_weight: f64,
        w_weight: f64,
        expected_distance: f64
    ) {
        let g1 = helper_mock_genome(s1);
        let g2 = helper_mock_genome(s2);

        let actual_distance = genome_distance(g1.synapse_genes(), g2.synapse_genes(), d_weight, e_weight, w_weight);

        assert!(
            (actual_distance - expected_distance).abs() < f64::EPSILON,
            "Expected {}, got {}", expected_distance, actual_distance
        )
    }

    //////////////////////////////
    // # GENOME CROSSOVER TESTS //
    //////////////////////////////

    /// node_nums = [0, 1, 3]
    /// inno_nums = [0, 6, 7, 10]
    fn helper_base_genome(id: usize) -> Genome {
        let neuron_genes: Vec<NeuronGene> = vec![
            NeuronGene { node_name: 0, neuron_type: NeuronType::Sensory(0) },
            NeuronGene { node_name: 1, neuron_type: NeuronType::Muscular(0) },
            NeuronGene { node_name: 3, neuron_type: NeuronType::Inter() },
        ];

        let synapse_genes: Vec<SynapseGene> = vec![
            SynapseGene{
                src_id: 0,
                tgt_id: 1,
                weight: 10.0,
                inno_num: 0,
                enabled: true,
            },
            SynapseGene{
                src_id: 0,
                tgt_id: 3,
                weight: 5.0,
                inno_num: 6,
                enabled: true,
            },
            SynapseGene{
                src_id: 3,
                tgt_id: 1,
                weight: 5.0,
                inno_num: 7,
                enabled: true,
            },
            SynapseGene{
                src_id: 1,
                tgt_id: 3,
                weight: 5.0,
                inno_num: 10,
                enabled: true,
            },
        ];

        Genome::new(id, 0, neuron_genes, synapse_genes).expect(
            "This test genome should be valid."
        )
    }

    /// node_nums = [0, 1, 3, 4]
    /// inno_nums = [0, 6, 7, 10, 11, 12]
    fn helper_excess_genome(id: usize) -> Genome {
        let neuron_genes: Vec<NeuronGene> = vec![
            NeuronGene { node_name: 0, neuron_type: NeuronType::Sensory(0) },
            NeuronGene { node_name: 1, neuron_type: NeuronType::Muscular(0) },
            NeuronGene { node_name: 3, neuron_type: NeuronType::Inter() },
            NeuronGene { node_name: 4, neuron_type: NeuronType::Inter() }
        ];

        let synapse_genes: Vec<SynapseGene> = vec![
            SynapseGene{
                src_id: 0,
                tgt_id: 1,
                weight: 20.0,
                inno_num: 0,
                enabled: false,
            },
            SynapseGene{
                src_id: 0,
                tgt_id: 3,
                weight: 5.0,
                inno_num: 6,
                enabled: true,
            },
            SynapseGene{
                src_id: 3,
                tgt_id: 1,
                weight: 5.0,
                inno_num: 7,
                enabled: true,
            },
            SynapseGene{
                src_id: 1,
                tgt_id: 3,
                weight: 5.0,
                inno_num: 10,
                enabled: false,
            },
            SynapseGene{
                src_id: 1,
                tgt_id: 4,
                weight: 5.0,
                inno_num: 11,
                enabled: true,
            },
            SynapseGene{
                src_id: 4,
                tgt_id: 3,
                weight: 5.0,
                inno_num: 12,
                enabled: true,
            },
        ];

        Genome::new(id, 0, neuron_genes, synapse_genes).expect(
            "This test genome should be valid."
        )
    }

    /// node_nums = [0, 1, 2, 3]
    /// inno_nums = [0, 2, 3, 6, 7, 10]
    fn helper_disjoint_genome(id: usize) -> Genome {
        let neuron_genes: Vec<NeuronGene> = vec![
            NeuronGene { node_name: 0, neuron_type: NeuronType::Sensory(0) },
            NeuronGene { node_name: 1, neuron_type: NeuronType::Muscular(0) },
            NeuronGene { node_name: 2, neuron_type: NeuronType::Muscular(0) },
            NeuronGene { node_name: 3, neuron_type: NeuronType::Inter() },
        ];

        let synapse_genes: Vec<SynapseGene> = vec![
            SynapseGene{
                src_id: 0,
                tgt_id: 1,
                weight: 5.0,
                inno_num: 0,
                enabled: false,
            },
            SynapseGene {
                src_id: 0,
                tgt_id: 2,
                weight: 5.0,
                inno_num: 2,
                enabled: true,
            },
            SynapseGene {
                src_id: 2,
                tgt_id: 1,
                weight: 5.0,
                inno_num: 3,
                enabled: true,
            },
            SynapseGene{
                src_id: 0,
                tgt_id: 3,
                weight: 5.0,
                inno_num: 6,
                enabled: true,
            },
            SynapseGene{
                src_id: 3,
                tgt_id: 1,
                weight: 5.0,
                inno_num: 7,
                enabled: true,
            },
            SynapseGene{
                src_id: 1,
                tgt_id: 3,
                weight: 5.0,
                inno_num: 10,
                enabled: true,
            },
        ];

        Genome::new(id, 0, neuron_genes, synapse_genes).expect(
            "This test genome should be valid."
        )
    }

    #[test_case(helper_base_genome(0), helper_base_genome(1), 10.0, 5.0, vec![0, 1, 3], vec![0, 6, 7, 10];
        "Matching homology, one dominant")]
    #[test_case(helper_base_genome(0), helper_base_genome(1), 10.0, 10.0, vec![0, 1, 3], vec![0, 6, 7, 10];
        "Matching homology, both dominant")]
    #[test_case(helper_excess_genome(0), helper_base_genome(1), 10.0, 5.0, vec![0, 1, 3, 4], vec![0, 6, 7, 10, 11, 12];
        "Excess homology, excess dominant")]
    #[test_case(helper_excess_genome(0), helper_base_genome(1), 5.0, 10.0, vec![0, 1, 3], vec![0, 6, 7, 10];
        "Excess homology, base dominant")]
    #[test_case(helper_excess_genome(0), helper_base_genome(1), 10.0, 10.0, vec![0, 1, 3, 4], vec![0, 6, 7, 10, 11, 12];
        "Excess homology, both dominant")]
    #[test_case(helper_disjoint_genome(0), helper_base_genome(1), 10.0, 5.0, vec![0, 1, 2, 3], vec![0, 2, 3, 6, 7, 10];
        "Disjoint homology, disjoint dominant")]
    #[test_case(helper_disjoint_genome(0), helper_base_genome(1), 5.0, 10.0, vec![0, 1, 3], vec![0, 6, 7, 10];
        "Disjoint homology, base dominant")]
    #[test_case(helper_disjoint_genome(0), helper_base_genome(1), 10.0, 10.0, vec![0, 1, 2, 3], vec![0, 2, 3, 6, 7, 10];
        "Disjoint homology, both dominant")]
    #[test_case(helper_disjoint_genome(0), helper_excess_genome(1), 10.0, 5.0, vec![0, 1, 2, 3], vec![0, 2, 3, 6, 7, 10];
        "Disjoint w/ Excess homology, disjoint dominant")]
    #[test_case(helper_disjoint_genome(0), helper_excess_genome(1), 5.0, 10.0, vec![0, 1, 3, 4], vec![0, 6, 7, 10, 11, 12];
        "Disjoint w/ Excess homology, excess dominant")]
    #[test_case(helper_disjoint_genome(0), helper_excess_genome(1), 10.0, 10.0, vec![0, 1, 2, 3, 4], vec![0, 2, 3, 6, 7, 10, 11, 12];
        "Disjoint w/ Excess homology, both dominant")]
    #[test_case(helper_excess_genome(0), helper_disjoint_genome(1), 10.0, 10.0, vec![0, 1, 2, 3, 4], vec![0, 2, 3, 6, 7, 10, 11, 12];
        "Excess w/ Disjoint homology, both dominant")]
    fn test_crossover(genome1: Genome, genome2: Genome, fitness1: f64, fitness2: f64, node_result: Vec<usize>, synapse_result: Vec<usize>) {
        let (neuron_genes, synapse_genes, _) = genome_crossover(&genome1, &genome2, fitness1, fitness2, 0.75);

        let actual_nodes: Vec<usize> = neuron_genes.iter().map(|n| n.node_name).collect();
        let actual_innos: Vec<usize> = synapse_genes.iter().map(|s| s.inno_num).collect();

        assert_eq!(actual_nodes, node_result, "Neuron nodes do not match expectations");
        assert_eq!(actual_innos, synapse_result, "Synapse innovation numbers do not match expectations");

        assert!(Genome::new(2, 0, neuron_genes, synapse_genes).is_ok(), "Two valid genomes crossing over should create a valid genome.")
    }

    #[test_case(10.0, 5.0, true; "Higher weight dominant, one enable dominant")]
    #[test_case(5.0, 10.0, true; "Higher weight recessive, one enable recessive")]
    #[test_case(5.0, 5.0, true; "Both dominant, one enable")]
    #[test_case(10.0, 5.0, false; "Higher weight dominant, no enable")]
    #[test_case(5.0, 10.0, false; "Higher weight recessive, no enable")]
    #[test_case(5.0, 5.0, false; "Both dominant, no enable")]
    fn test_crossover_inherit_probabilities(fitness1: f64, fitness2: f64, enable1: bool) {
        let mut genome1 = helper_base_genome(0);
        let mut genome2 = helper_base_genome(1);

        let iterations = 10_000;
        let tolerance= 0.02;

        let expected_weight_prob = 0.50;
        let expected_disabled_prob = 0.75;

        genome1.synapse_genes[0].weight = 20.0;
        genome1.synapse_genes[0].enabled = enable1;

        genome2.synapse_genes[0].weight = 10.0;
        genome2.synapse_genes[0].enabled = false;

        let mut p1_weight_count = 0;
        let mut disabled_count = 0;

        for _ in 0..iterations {
            let (_, child_synapses, _) = genome_crossover(&genome1, &genome2, fitness1, fitness2, expected_disabled_prob);
            let target_gene = &child_synapses[0];

            if (target_gene.weight - 20.0).abs() < f64::EPSILON {
                p1_weight_count += 1;
            }
            else {
                assert!((target_gene.weight - 10.0).abs() < f64::EPSILON, "Children should not inherit a weight that belonged to neither parent!");
            }

            if !target_gene.enabled {
                disabled_count += 1;
            }
        }

        let observed_weight_prob = p1_weight_count as f64 / iterations as f64;
        let observed_disabled_prob = disabled_count as f64 / iterations as f64;

        assert!((observed_weight_prob - expected_weight_prob).abs() < tolerance,
                "Weight inheritance is not within expected confidence of 50%");
        assert!((observed_disabled_prob - expected_disabled_prob).abs() < tolerance,
                "Disable inheritance is not within expected confidence of input of 75%");
    }

    #[test_case(10.0, 5.0; "One dominant")]
    #[test_case(5.0, 5.0; "Both dominant")]
    fn test_crossover_both_enabled(fitness1: f64, fitness2: f64) {
        let mut genome1 = helper_base_genome(0);
        let mut genome2 = helper_base_genome(1);

        let iterations = 10_000;
        let tolerance= 0.02;

        let expected_disabled_prob = 0.75;

        genome1.synapse_genes[0].enabled = true;

        genome2.synapse_genes[0].enabled = true;

        for _ in 0..iterations {
            let (_, child_synapses, _) = genome_crossover(&genome1, &genome2, fitness1, fitness2, expected_disabled_prob);
            let target_gene = &child_synapses[0];

            assert_eq!(target_gene.enabled, true, "Children should not inherit a gene both parents have enabled as disabled.")
        }
    }
}