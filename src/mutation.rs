use rand::distr::Distribution;
use std::collections::HashMap;
use rand::distr::Uniform;
use rand::prelude::*;
use crate::genome::{NeuronGene, NeuronType, SynapseGene};

pub struct MutationParameters {
    mutate_edge_weight_prob: f64,
    mutate_edge_random_weight_prob: f64,
    mutate_enable_prob: f64,
    mutate_disable_prob: f64,
    mutation_power: f64,
    mutate_new_edge_prob: f64,
    mutate_new_node_prob: f64,

    edge_weight_cap: f64,
    edge_new_weight_cap: f64,
}
impl MutationParameters {
    pub fn default_params() -> Self {
        Self{
            mutate_edge_weight_prob: 0.8,
            mutate_edge_random_weight_prob: 0.1,
            mutate_enable_prob: 0.05,
            mutate_disable_prob: 0.005,
            mutation_power: 2.5,
            mutate_new_edge_prob: 0.05,
            mutate_new_node_prob: 0.03,
            edge_weight_cap: 8.0,
            edge_new_weight_cap: 1.0,
        }
    }
}

pub fn mutate_no_structure_change<R: Rng>(rng: &mut R,
                                          params: &MutationParameters,
                                          mut synapse_genes: Vec<SynapseGene>)
    -> Vec<SynapseGene> {
    let weight_dist = Uniform::new_inclusive(-params.mutation_power, params.mutation_power)
        .expect("Mutation power should be strictly positive and finite.");

    // Mutate weights on individual synapse genes.
    for gene in synapse_genes.iter_mut() {
        if rng.random_bool(params.mutate_edge_weight_prob) {
            mutate_edge_weight(rng, params, &weight_dist, gene);
        }
    }

    synapse_genes
}

pub fn mutate<R: Rng>(rng: &mut R,
                      params: &MutationParameters,
                      next_inno_index: &mut usize,
                      next_node_index: &mut usize,
                      new_innos: &mut HashMap<(usize, usize), usize>,
                      new_nodes: &mut HashMap<usize, usize>,
                      mut neuron_genes: Vec<NeuronGene>,
                      mut synapse_genes: Vec<SynapseGene>,)
    -> (Vec<NeuronGene>, Vec<SynapseGene>) {
    let weight_dist = Uniform::new_inclusive(-params.mutation_power, params.mutation_power)
        .expect("Mutation power should be strictly positive and finite.");

    // Mutate weights on individual synapse genes.
    let num_genes = synapse_genes.len() as f64;
    for gene in synapse_genes.iter_mut() {
        if rng.random_bool(params.mutate_edge_weight_prob) {
            mutate_edge_weight(rng, params, &weight_dist, gene);
        }

        if gene.enabled {
            if rng.random_bool(params.mutate_disable_prob / num_genes) {
                gene.enabled = false;
            }
        } else {
            if rng.random_bool(params.mutate_enable_prob / num_genes) {
                gene.enabled = true;
            }
        }
    }

    // Mutate a new node.
    // This one is guaranteed to succeed.
    if rng.random_bool(params.mutate_new_node_prob) {
        mutate_new_node(rng, next_inno_index, next_node_index, new_innos, new_nodes, &mut neuron_genes, &mut synapse_genes);
    }

    // Mutate a new edge.
    // Note that this is done with rejection sampling, and so it is not guaranteed to always
    // succeed. There are more clever ways to do this if synapse_genes was better organized, but
    // that defeats NEAT's homology requirements, which leaves us stuck with the lazy solution
    // unless an *extra* clever solution is found.
    if rng.random_bool(params.mutate_new_edge_prob / num_genes) {
        mutate_new_edge(rng, params, next_inno_index, new_innos, &mut neuron_genes, &mut synapse_genes);
    }

    (neuron_genes, synapse_genes)
}

#[inline]
fn mutate_edge_weight<R: Rng>(rng: &mut R,
                              params: &MutationParameters,
                              weight_dist: &Uniform<f64>,
                              gene: &mut SynapseGene) {
    let nudge = weight_dist.sample(rng);
    if rng.random_bool(params.mutate_edge_random_weight_prob) {
        gene.weight = nudge;
    } else {
        gene.weight += nudge;
        gene.weight = gene.weight.clamp(-params.edge_weight_cap, params.edge_weight_cap);
    }
}

#[inline]
fn mutate_new_edge<R: Rng>(rng: &mut R,
                           params: &MutationParameters,
                           next_inno_index: &mut usize,
                           new_innos: &mut HashMap<(usize, usize), usize>,
                           neuron_genes: &[NeuronGene],
                           synapse_genes: &mut Vec<SynapseGene>) {
    for _ in 0..20 {
        let src_id: usize = rng.random_range(0..neuron_genes.len());
        let tgt_id: usize = rng.random_range(0..neuron_genes.len());

        // Reject if the target is a sensory type neuron.
        if matches!(neuron_genes[tgt_id].neuron_type, NeuronType::Sensory(_)) {
            continue;
        }
        // TODO:
        // If we knew the number of sensory neurons already present, we can guarantee we do not
        // choose a sensory neuron by tgt_id displacement. Perhaps worth counting earlier in the
        // mutation cycle.

        let src_name: usize = neuron_genes[src_id].node_name;
        let tgt_name: usize = neuron_genes[tgt_id].node_name;

        // Reject if this edge appears elsewhere.
        if synapse_genes.iter().any(
            |gene| gene.src_name == src_name && gene.tgt_name == tgt_name
        ) {
            continue;
        }

        let inno_num = *new_innos.entry((src_name, tgt_name))
            .or_insert_with(|| {
                let new_num = *next_inno_index;
                *next_inno_index += 1;
                new_num
            });

        let weight: f64 = rng.random_range(-params.edge_new_weight_cap..=params.edge_new_weight_cap);
        let new_synapse: SynapseGene = SynapseGene {
            src_name,
            tgt_name,
            weight,
            inno_num,
            enabled: true,
        };
        synapse_genes.push(new_synapse);
        break;
    }
}

#[inline]
fn mutate_new_node<R: Rng>(rng: &mut R,
                           next_inno_index: &mut usize,
                           next_node_index: &mut usize,
                           new_innos: &mut HashMap<(usize, usize), usize>,
                           new_nodes: &mut HashMap<usize, usize>,
                           neuron_genes: &mut Vec<NeuronGene>,
                           synapse_genes: &mut Vec<SynapseGene>) {

    // Choose enabled gene to split
    // TODO:
    // If we knew the number of disabled genes already present, we can guarantee we do not
    // choose a disabled gene by displacement. Perhaps worth counting earlier in the
    // mutation cycle.
    let enabled_count = synapse_genes.iter().filter(|g| g.enabled).count();
    let target_idx = if enabled_count > 0 {
        let pick = rng.random_range(0..enabled_count);
        synapse_genes.iter()
            .enumerate()
            .filter(|(_, gene)| gene.enabled)
            .map(|(idx, _)| idx)
            .nth(pick)
    } else {
        None
    };

    if let Some(target_idx) = target_idx {
        let old_synapse = synapse_genes[target_idx].clone();
        synapse_genes[target_idx].enabled = false;

        let is_new_split = !new_nodes.contains_key(&old_synapse.inno_num);

        let new_node_id = *new_nodes.entry(old_synapse.inno_num).or_insert_with(|| {
            let id = *next_node_index;
            *next_node_index += 1;
            id
        });

        let (inno1, inno2) = if is_new_split {
            let i1 = *next_inno_index;
            let i2 = *next_inno_index + 1;
            *next_inno_index += 2;

            new_innos.insert((old_synapse.src_name, new_node_id), i1);
            new_innos.insert((new_node_id, old_synapse.tgt_name), i2);

            (i1, i2)
        } else {
            let i1 = *new_innos.get(&(old_synapse.src_name, new_node_id))
                .expect("New node innovation already evolved.");
            let i2 = i1 + 1;
            (i1, i2)
        };

        synapse_genes.push(SynapseGene {
            src_name: old_synapse.src_name,
            tgt_name: new_node_id,
            weight: 1.0,
            inno_num: inno1,
            enabled: old_synapse.enabled,
        });

        synapse_genes.push(SynapseGene {
            src_name: new_node_id,
            tgt_name: old_synapse.tgt_name,
            weight: old_synapse.weight,
            inno_num: inno2,
            enabled: true,
        });

        neuron_genes.push(NeuronGene {
            node_name: new_node_id,
            neuron_type: NeuronType::Inter()
        })
    }
}


#[cfg(test)]
mod test_mutation {
    use super::*;
    use rand::distr::Uniform;
    use test_case::test_case;
    use crate::genome::{SynapseGene};
    use crate::mock_rng::MockRng;

    fn produce_example_genes() -> (Vec<NeuronGene>, Vec<SynapseGene>) {
        let neuron_genes: Vec<NeuronGene> = vec![NeuronGene{ node_name: 0, neuron_type: NeuronType::Sensory(0)},
                                                 NeuronGene{ node_name: 4, neuron_type: NeuronType::Muscular(0)},
                                                 NeuronGene{ node_name: 7, neuron_type: NeuronType::Inter()},
                                                 NeuronGene{ node_name: 12, neuron_type: NeuronType::Inter()}];

        let synapse_genes: Vec<SynapseGene> = vec![
            SynapseGene{
                src_name: 0,
                tgt_name: 4,
                weight: 1.0,
                inno_num: 0,
                enabled: false,
            },
            SynapseGene{
                src_name: 0,
                tgt_name: 7,
                weight: 1.0,
                inno_num: 8,
                enabled: true,
            },
            SynapseGene{
                src_name: 7,
                tgt_name: 4,
                weight: 1.0,
                inno_num: 9,
                enabled: true,
            },
            SynapseGene{
                src_name: 0,
                tgt_name: 12,
                weight: 2.0,
                inno_num: 40,
                enabled: true,
            },
            SynapseGene{
                src_name: 12,
                tgt_name: 4,
                weight: 2.0,
                inno_num: 41,
                enabled: true,
            }
        ];
        (neuron_genes, synapse_genes)
    }

    //////////////////////////////
    // MUTATE_EDGE_WEIGHT TESTS //
    //////////////////////////////

    #[test_case(7.5, 1.0, 8.0; "overflow positive 1")]
    #[test_case(6.5, 2.0, 8.0; "overflow positive 2")]
    #[test_case(5.0, 2.0, 7.0; "normal positive 1")]
    #[test_case(6.0, 1.0, 7.0; "normal positive 2")]
    #[test_case(-7.5, -1.0, -8.0; "underflow negative 1")]
    #[test_case(-6.5, -2.0, -8.0; "underflow negative 2")]
    #[test_case(-5.0, -2.0, -7.0; "normal negative 1")]
    #[test_case(-6.0, -1.0, -7.0; "normal negative 2")]
    fn test_mutate_edge_weight_clamp(existing_weight: f64, nudge: f64, final_weight: f64) {
        let mut params = MutationParameters::default_params();
        params.mutation_power = 2.5;
        params.edge_weight_cap = 8.0;

        let distribution = Uniform::new_inclusive(-2.5, 2.5).unwrap();

        let mut mock_rng = MockRng::new();
        mock_rng.push_uniform_float_inclusive(nudge, -2.5, 2.5);
        mock_rng.push_bool(false);

        let mut gene = SynapseGene{
            src_name: 0,
            tgt_name: 1,
            weight: existing_weight,
            inno_num: 0,
            enabled: false,
        };

        mutate_edge_weight(&mut mock_rng, &params, &distribution, &mut gene);

        assert!((gene.weight - final_weight).abs() <= f64::EPSILON);
    }

    #[test_case(-7.5, 1.0; "1")]
    #[test_case(6.5, 2.0; "2")]
    #[test_case(-5.0, 2.0; "3")]
    #[test_case(6.0, 0.1; "4")]
    fn test_mutate_edge_weight_equals_nudge(existing_weight: f64, nudge: f64) {
        let mut params = MutationParameters::default_params();
        params.mutation_power = 2.5; // 2.5
        params.edge_weight_cap = 8.0; // 8.0

        let distribution = Uniform::new_inclusive(-2.5, 2.5).unwrap();

        let mut mock_rng = MockRng::new();
        let expected = mock_rng.push_uniform_float_inclusive(nudge, -2.5, 2.5);
        mock_rng.push_bool(true);

        let mut gene = SynapseGene{
            src_name: 0,
            tgt_name: 1,
            weight: existing_weight,
            inno_num: 0,
            enabled: false,
        };

        mutate_edge_weight(&mut mock_rng, &params, &distribution, &mut gene);

        assert_eq!(gene.weight, expected);
    }

    ///////////////////////////
    // MUTATE_NEW_EDGE TESTS //
    ///////////////////////////

    #[test]
    fn test_mutate_new_edge_new_inno() {
        let mut params = MutationParameters::default_params();
        params.edge_new_weight_cap = 1.0;

        let mut next_inno_index: usize = 60;

        let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
        new_innos.insert((7, 12), 59);

        let (neuron_genes, mut synapse_genes) = produce_example_genes();

        let mut mock_rng = MockRng::new();
        mock_rng.push_index(3, 4);
        mock_rng.push_index(2, 4);
        let weight = mock_rng.push_range_float(0.5, -1.0, 1.0);

        mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &*neuron_genes, &mut synapse_genes);

        assert_eq!(next_inno_index, 61);

        assert_eq!(new_innos.len(), 2);
        assert!(new_innos.contains_key(&(12, 7)));
        assert_eq!(*new_innos.get(&(12, 7)).unwrap(), 60);

        let expected_synapse = SynapseGene{
            src_name: 12,
            tgt_name: 7,
            weight,
            inno_num: 60,
            enabled: true,
        };

        assert_eq!(synapse_genes.len(), 6);

        let produced_synapse = &synapse_genes[5];

        assert_eq!(expected_synapse.src_name, produced_synapse.src_name);
        assert_eq!(expected_synapse.tgt_name, produced_synapse.tgt_name);
        assert_eq!(expected_synapse.weight, produced_synapse.weight);
        assert_eq!(expected_synapse.inno_num, produced_synapse.inno_num);
        assert_eq!(expected_synapse.enabled, produced_synapse.enabled);
    }

    #[test]
    fn test_mutate_new_edge_repeat_inno() {
        let mut params = MutationParameters::default_params();
        params.edge_new_weight_cap = 1.0;

        let mut next_inno_index: usize = 60;

        let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
        new_innos.insert((7, 12), 59);

        let (neuron_genes, mut synapse_genes) = produce_example_genes();

        let mut mock_rng = MockRng::new();
        mock_rng.push_index(2, 4);
        mock_rng.push_index(3, 4);
        let weight = mock_rng.push_range_float(0.5, -1.0, 1.0);

        mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &*neuron_genes, &mut synapse_genes);

        assert_eq!(next_inno_index, 60);

        assert_eq!(new_innos.len(), 1);

        let expected_synapse = SynapseGene{
            src_name: 7,
            tgt_name: 12,
            weight,
            inno_num: 59,
            enabled: true,
        };

        assert_eq!(synapse_genes.len(), 6);

        let produced_synapse = &synapse_genes[5];

        assert_eq!(expected_synapse.src_name, produced_synapse.src_name);
        assert_eq!(expected_synapse.tgt_name, produced_synapse.tgt_name);
        assert_eq!(expected_synapse.weight, produced_synapse.weight);
        assert_eq!(expected_synapse.inno_num, produced_synapse.inno_num);
        assert_eq!(expected_synapse.enabled, produced_synapse.enabled);
    }

    #[test]
    fn test_mutate_new_edge_no_sensory_targets() {
        let mut params = MutationParameters::default_params();
        params.edge_new_weight_cap = 1.0;

        let mut next_inno_index: usize = 60;

        let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
        new_innos.insert((7, 12), 59);

        let (neuron_genes, mut synapse_genes) = produce_example_genes();

        let mut mock_rng = MockRng::new();
        mock_rng.push_index(3, 4);
        mock_rng.push_index(0, 4);
        mock_rng.push_index(3, 4);
        mock_rng.push_index(2, 4);
        mock_rng.push_range_float(0.5, -1.0, 1.0);

        mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &*neuron_genes, &mut synapse_genes);

        assert!(!new_innos.contains_key(&(12, 0)));

        let produced_synapse = &synapse_genes[5];

        assert_ne!(produced_synapse.tgt_name, 0);
    }

    #[test]
    fn test_mutate_new_edge_no_repeat_edges() {
        let mut params = MutationParameters::default_params();
        params.edge_new_weight_cap = 1.0;

        let mut next_inno_index: usize = 60;

        let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
        new_innos.insert((7, 12), 59);

        let (neuron_genes, mut synapse_genes) = produce_example_genes();

        let mut mock_rng = MockRng::new();
        mock_rng.push_index(0, 4);
        mock_rng.push_index(1, 4);
        mock_rng.push_index(3, 4);
        mock_rng.push_index(2, 4);
        mock_rng.push_range_float(0.5, -1.0, 1.0);

        mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &*neuron_genes, &mut synapse_genes);

        assert!(!new_innos.contains_key(&(0, 4)));

        let produced_synapse = &synapse_genes[5];

        assert_ne!(produced_synapse.src_name, 0);
        assert_ne!(produced_synapse.tgt_name, 4);
    }

    fn populate_rejections(mock_rng: &mut MockRng) {
        mock_rng.push_index(3, 4); mock_rng.push_index(0, 4);
        mock_rng.push_index(3, 4); mock_rng.push_index(0, 4);
        mock_rng.push_index(3, 4); mock_rng.push_index(0, 4);
        mock_rng.push_index(3, 4); mock_rng.push_index(0, 4);
        mock_rng.push_index(3, 4); mock_rng.push_index(0, 4);
        mock_rng.push_index(3, 4); mock_rng.push_index(0, 4);
        mock_rng.push_index(3, 4); mock_rng.push_index(0, 4);
        mock_rng.push_index(3, 4); mock_rng.push_index(0, 4);
        mock_rng.push_index(3, 4); mock_rng.push_index(0, 4);
        mock_rng.push_index(3, 4); mock_rng.push_index(0, 4);
        mock_rng.push_index(0, 4); mock_rng.push_index(1, 4);
        mock_rng.push_index(0, 4); mock_rng.push_index(1, 4);
        mock_rng.push_index(0, 4); mock_rng.push_index(1, 4);
        mock_rng.push_index(0, 4); mock_rng.push_index(2, 4);
        mock_rng.push_index(0, 4); mock_rng.push_index(2, 4);
        mock_rng.push_index(0, 4); mock_rng.push_index(2, 4);
        mock_rng.push_index(2, 4); mock_rng.push_index(1, 4);
        mock_rng.push_index(2, 4); mock_rng.push_index(1, 4);
        mock_rng.push_index(2, 4); mock_rng.push_index(1, 4);
    }

    #[test]
    fn test_mutate_new_edge_19_rejections_new_inno() {
        let mut params = MutationParameters::default_params();
        params.edge_new_weight_cap = 1.0;

        let mut next_inno_index: usize = 60;

        let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
        new_innos.insert((7, 12), 59);

        let (neuron_genes, mut synapse_genes) = produce_example_genes();

        let mut mock_rng = MockRng::new();
        populate_rejections(&mut mock_rng);
        mock_rng.push_index(3, 4);
        mock_rng.push_index(2, 4);
        let weight = mock_rng.push_range_float(0.5, -1.0, 1.0);

        mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &*neuron_genes, &mut synapse_genes);

        assert_eq!(next_inno_index, 61);

        assert_eq!(new_innos.len(), 2);
        assert!(new_innos.contains_key(&(12, 7)));
        assert_eq!(*new_innos.get(&(12, 7)).unwrap(), 60);

        let expected_synapse = SynapseGene{
            src_name: 12,
            tgt_name: 7,
            weight,
            inno_num: 60,
            enabled: true,
        };

        assert_eq!(synapse_genes.len(), 6);

        let produced_synapse = &synapse_genes[5];

        assert_eq!(expected_synapse.src_name, produced_synapse.src_name);
        assert_eq!(expected_synapse.tgt_name, produced_synapse.tgt_name);
        assert_eq!(expected_synapse.weight, produced_synapse.weight);
        assert_eq!(expected_synapse.inno_num, produced_synapse.inno_num);
        assert_eq!(expected_synapse.enabled, produced_synapse.enabled);
    }

    #[test]
    fn test_mutate_new_edge_19_rejections_repeat_inno() {
        let mut params = MutationParameters::default_params();
        params.edge_new_weight_cap = 1.0;

        let mut next_inno_index: usize = 60;

        let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
        new_innos.insert((7, 12), 59);

        let (neuron_genes, mut synapse_genes) = produce_example_genes();

        let mut mock_rng = MockRng::new();
        populate_rejections(&mut mock_rng);
        mock_rng.push_index(2, 4);
        mock_rng.push_index(3, 4);
        let weight = mock_rng.push_range_float(0.5, -1.0, 1.0);

        mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &*neuron_genes, &mut synapse_genes);

        assert_eq!(next_inno_index, 60);
        assert_eq!(new_innos.len(), 1);

        let expected_synapse = SynapseGene{
            src_name: 7,
            tgt_name: 12,
            weight,
            inno_num: 59,
            enabled: true,
        };

        assert_eq!(synapse_genes.len(), 6);

        let produced_synapse = &synapse_genes[5];

        assert_eq!(expected_synapse.src_name, produced_synapse.src_name);
        assert_eq!(expected_synapse.tgt_name, produced_synapse.tgt_name);
        assert_eq!(expected_synapse.weight, produced_synapse.weight);
        assert_eq!(expected_synapse.inno_num, produced_synapse.inno_num);
        assert_eq!(expected_synapse.enabled, produced_synapse.enabled);
    }

    #[test]
    fn test_mutate_new_edge_rejection_limit_reached() {
        let mut params = MutationParameters::default_params();
        params.edge_new_weight_cap = 1.0;

        let mut next_inno_index: usize = 60;

        let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
        new_innos.insert((7, 12), 59);

        let (neuron_genes, mut synapse_genes) = produce_example_genes();

        let mut mock_rng = MockRng::new();
        populate_rejections(&mut mock_rng);
        mock_rng.push_index(3, 4);
        mock_rng.push_index(1, 4);
        mock_rng.push_range_float(0.5, -1.0, 1.0);

        mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &*neuron_genes, &mut synapse_genes);

        assert_eq!(next_inno_index, 60);
        assert_eq!(new_innos.len(), 1);

        assert_eq!(synapse_genes.len(), 5);
    }

    ///////////////////////////
    // MUTATE_NEW_NODE TESTS //
    ///////////////////////////

    #[test]
    fn test_mutate_new_node_new_inno() {
        let mut next_inno_index: usize = 60;
        let mut next_node_index: usize = 15;

        let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
        new_innos.insert((0, 14), 57);
        new_innos.insert((14, 12), 58);
        new_innos.insert((7, 12), 59);

        let mut new_nodes: HashMap<usize, usize> = HashMap::new();
        new_nodes.insert(40, 14);

        let (mut neuron_genes, mut synapse_genes) = produce_example_genes();


        let mut mock_rng = MockRng::new();
        mock_rng.push_index(3, 4);

        mutate_new_node(&mut mock_rng, &mut next_inno_index, &mut next_node_index, &mut new_innos, &mut new_nodes, &mut neuron_genes, &mut synapse_genes);

        assert_eq!(next_inno_index, 62);
        assert_eq!(next_node_index, 16);

        assert_eq!(new_innos.len(), 5);
        assert!(new_innos.contains_key(&(12, 15)));
        assert_eq!(*new_innos.get(&(12, 15)).unwrap(), 60);
        assert!(new_innos.contains_key(&(15, 4)));
        assert_eq!(*new_innos.get(&(15, 4)).unwrap(), 61);

        assert_eq!(new_nodes.len(), 2);
        assert!(new_nodes.contains_key(&41));
        assert_eq!(*new_nodes.get(&41).unwrap(), 15);

        assert_eq!(neuron_genes.len(), 5);
        let new_neuron = &neuron_genes[4];
        assert_eq!(new_neuron.node_name, 15);
        assert!(matches!(new_neuron.neuron_type, NeuronType::Inter()));

        assert_eq!(synapse_genes.len(), 7);
        let expected_new_synapse = SynapseGene{
            src_name: 12,
            tgt_name: 15,
            weight: 1.0,
            inno_num: 60,
            enabled: true,
        };
        let actual_new_synapse = &synapse_genes[5];
        assert_eq!(expected_new_synapse.src_name, actual_new_synapse.src_name);
        assert_eq!(expected_new_synapse.tgt_name, actual_new_synapse.tgt_name);
        assert_eq!(expected_new_synapse.weight, actual_new_synapse.weight);
        assert_eq!(expected_new_synapse.inno_num, actual_new_synapse.inno_num);
        assert_eq!(expected_new_synapse.enabled, actual_new_synapse.enabled);

        let expected_new_synapse = SynapseGene{
            src_name: 15,
            tgt_name: 4,
            weight: 2.0,
            inno_num: 61,
            enabled: true,
        };
        let actual_new_synapse = &synapse_genes[6];
        assert_eq!(expected_new_synapse.src_name, actual_new_synapse.src_name);
        assert_eq!(expected_new_synapse.tgt_name, actual_new_synapse.tgt_name);
        assert_eq!(expected_new_synapse.weight, actual_new_synapse.weight);
        assert_eq!(expected_new_synapse.inno_num, actual_new_synapse.inno_num);
        assert_eq!(expected_new_synapse.enabled, actual_new_synapse.enabled);

        assert!(!synapse_genes[4].enabled);
    }

    #[test]
    fn test_mutate_new_node_repeat_inno() {
        let mut next_inno_index: usize = 60;
        let mut next_node_index: usize = 15;

        let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
        new_innos.insert((0, 14), 57);
        new_innos.insert((14, 12), 58);
        new_innos.insert((7, 12), 59);

        let mut new_nodes: HashMap<usize, usize> = HashMap::new();
        new_nodes.insert(40, 14);

        let (mut neuron_genes, mut synapse_genes) = produce_example_genes();
        synapse_genes[0].enabled = false;
        synapse_genes[1].enabled = false;
        synapse_genes[2].enabled = false;
        synapse_genes[3].enabled = true;
        synapse_genes[4].enabled = false;

        let mut mock_rng = MockRng::new();
        mock_rng.push_index(2, 4);

        mutate_new_node(&mut mock_rng, &mut next_inno_index, &mut next_node_index, &mut new_innos, &mut new_nodes, &mut neuron_genes, &mut synapse_genes);

        assert_eq!(next_inno_index, 60);
        assert_eq!(next_node_index, 15);

        assert_eq!(new_innos.len(), 3);

        assert_eq!(new_nodes.len(), 1);

        assert_eq!(neuron_genes.len(), 5);
        let new_neuron = &neuron_genes[4];
        assert_eq!(new_neuron.node_name, 14);
        assert!(matches!(new_neuron.neuron_type, NeuronType::Inter()));

        assert_eq!(synapse_genes.len(), 7);
        let expected_new_synapse = SynapseGene{
            src_name: 0,
            tgt_name: 14,
            weight: 1.0,
            inno_num: 57,
            enabled: true,
        };
        let actual_new_synapse = &synapse_genes[5];
        assert_eq!(expected_new_synapse.src_name, actual_new_synapse.src_name);
        assert_eq!(expected_new_synapse.tgt_name, actual_new_synapse.tgt_name);
        assert_eq!(expected_new_synapse.weight, actual_new_synapse.weight);
        assert_eq!(expected_new_synapse.inno_num, actual_new_synapse.inno_num);
        assert_eq!(expected_new_synapse.enabled, actual_new_synapse.enabled);

        let expected_new_synapse = SynapseGene{
            src_name: 14,
            tgt_name: 12,
            weight: 2.0,
            inno_num: 58,
            enabled: true,
        };
        let actual_new_synapse = &synapse_genes[6];
        assert_eq!(expected_new_synapse.src_name, actual_new_synapse.src_name);
        assert_eq!(expected_new_synapse.tgt_name, actual_new_synapse.tgt_name);
        assert_eq!(expected_new_synapse.weight, actual_new_synapse.weight);
        assert_eq!(expected_new_synapse.inno_num, actual_new_synapse.inno_num);
        assert_eq!(expected_new_synapse.enabled, actual_new_synapse.enabled);

        assert!(!synapse_genes[4].enabled);
    }

    #[test]
    fn test_mutate_new_node_all_genes_disabled() {
        let mut next_inno_index: usize = 60;
        let mut next_node_index: usize = 15;

        let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
        new_innos.insert((0, 14), 57);
        new_innos.insert((14, 12), 58);
        new_innos.insert((7, 12), 59);

        let mut new_nodes: HashMap<usize, usize> = HashMap::new();
        new_nodes.insert(40, 14);

        let (mut neuron_genes, mut synapse_genes) = produce_example_genes();
        synapse_genes[0].enabled = false;
        synapse_genes[1].enabled = false;
        synapse_genes[2].enabled = false;
        synapse_genes[3].enabled = false;
        synapse_genes[4].enabled = false;

        let mut mock_rng = MockRng::new();
        mock_rng.push_index(2, 4);

        mutate_new_node(&mut mock_rng, &mut next_inno_index, &mut next_node_index, &mut new_innos, &mut new_nodes, &mut neuron_genes, &mut synapse_genes);

        assert_eq!(next_inno_index, 60);
        assert_eq!(next_node_index, 15);

        assert_eq!(new_innos.len(), 3);

        assert_eq!(new_nodes.len(), 1);

        assert_eq!(neuron_genes.len(), 4);

        assert_eq!(synapse_genes.len(), 5);
    }

    //////////////////
    // MUTATE TESTS //
    //////////////////

    #[test]
    fn test_mutation_can_toggle_genes() {
        let params = MutationParameters::default_params();

        let mut next_inno_index: usize = 60;
        let mut next_node_index: usize = 15;

        let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
        new_innos.insert((0, 14), 57);
        new_innos.insert((14, 12), 58);
        new_innos.insert((7, 12), 59);

        let mut new_nodes: HashMap<usize, usize> = HashMap::new();
        new_nodes.insert(40, 14);

        let (neuron_genes, synapse_genes) = produce_example_genes();

        let mut mock_rng = MockRng::new();
        mock_rng.push_bool(false);
        mock_rng.push_bool(true);
        mock_rng.push_bool(false);
        mock_rng.push_bool(true);
        mock_rng.push_bool(false);
        mock_rng.push_bool(false);
        mock_rng.push_bool(false);
        mock_rng.push_bool(false);
        mock_rng.push_bool(false);
        mock_rng.push_bool(false);
        mock_rng.push_bool(false);
        mock_rng.push_bool(false);

        let (_, synapse_genes) = mutate(&mut mock_rng, &params, &mut next_inno_index, &mut next_node_index, &mut new_innos, &mut new_nodes, neuron_genes, synapse_genes);

        assert!(synapse_genes[0].enabled);
        assert!(!synapse_genes[1].enabled);
        assert!(synapse_genes[2].enabled);
        assert!(synapse_genes[3].enabled);
        assert!(synapse_genes[4].enabled);
    }

    #[test]
    fn test_mutation_can_mutate_weights() {
        let mut params = MutationParameters::default_params();
        params.mutation_power = 2.5;
        params.edge_weight_cap = 8.0;

        let mut next_inno_index: usize = 60;
        let mut next_node_index: usize = 15;

        let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
        new_innos.insert((0, 14), 57);
        new_innos.insert((14, 12), 58);
        new_innos.insert((7, 12), 59);

        let mut new_nodes: HashMap<usize, usize> = HashMap::new();
        new_nodes.insert(40, 14);

        let (neuron_genes, synapse_genes) = produce_example_genes();

        let mut mock_rng = MockRng::new();
        // Weight 1
        mock_rng.push_bool(true);
        let weight_1 = mock_rng.push_uniform_float_inclusive(1.1, -2.5, 2.5) + 1.0;
        mock_rng.push_bool(false);
        mock_rng.push_bool(true);

        // Weight 2
        mock_rng.push_bool(true);
        let weight_2 = mock_rng.push_uniform_float_inclusive(2.3, -2.5, 2.5);
        mock_rng.push_bool(true);
        mock_rng.push_bool(false);

        mock_rng.push_bool(false);
        mock_rng.push_bool(true);

        mock_rng.push_bool(false);
        mock_rng.push_bool(false);

        mock_rng.push_bool(false);
        mock_rng.push_bool(false);

        mock_rng.push_bool(false);

        mock_rng.push_bool(false);

        let (_, synapse_genes) = mutate(&mut mock_rng, &params, &mut next_inno_index, &mut next_node_index, &mut new_innos, &mut new_nodes, neuron_genes, synapse_genes);

        assert!(synapse_genes[0].enabled);
        assert!(synapse_genes[1].enabled);
        assert!(!synapse_genes[2].enabled);
        assert!(synapse_genes[3].enabled);
        assert!(synapse_genes[4].enabled);

        assert_eq!(synapse_genes[0].weight, weight_1);
        assert_eq!(synapse_genes[1].weight, weight_2);
        assert_eq!(synapse_genes[2].weight, 1.0);
        assert_eq!(synapse_genes[3].weight, 2.0);
        assert_eq!(synapse_genes[4].weight, 2.0);
    }

    enum IntegrationSetting {
        NoChange,
        NewInno,
        OldInno,
    }
    use IntegrationSetting::*;

    #[test_case(NoChange, NoChange; "No Change")]
    #[test_case(NewInno, NoChange; "Novel New Edge")]
    #[test_case(OldInno, NoChange; "Repeat New Edge")]
    #[test_case(NoChange, NewInno; "Novel New Node")]
    #[test_case(NewInno, NewInno; "Novel New Edge, Novel New Node")]
    #[test_case(OldInno, NewInno; "Repeat New Edge, Novel New Node")]
    #[test_case(NoChange, OldInno; "Repeat New Node")]
    #[test_case(NewInno, OldInno; "Novel New Edge, Repeat New Node")]
    #[test_case(OldInno, OldInno; "Repeat New Edge, Repeat New Node")]
    fn test_mutation_integration(new_edge_setting: IntegrationSetting, new_node_setting: IntegrationSetting) {
        let params = MutationParameters::default_params();

        let mut next_inno_index: usize = 60;
        let mut expected_inno_index: usize = next_inno_index;
        let mut next_node_index: usize = 15;
        let mut expected_node_index: usize = next_node_index;

        let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
        new_innos.insert((0, 14), 57);
        new_innos.insert((14, 12), 58);
        new_innos.insert((7, 12), 59);
        let mut expected_new_innos: usize = 3;

        let mut new_nodes: HashMap<usize, usize> = HashMap::new();
        new_nodes.insert(40, 14);
        let mut expected_new_nodes: usize = 1;

        let (neuron_genes, synapse_genes) = produce_example_genes();
        let mut expected_neuron_genes: usize = 4;
        let mut expected_synapse_genes: usize = 5;

        let mut mock_rng = MockRng::new();
        // No change to existing synapses by weight/enable mutations
        mock_rng.push_bool(false);
        mock_rng.push_bool(false);
        mock_rng.push_bool(false);
        mock_rng.push_bool(false);
        mock_rng.push_bool(false);
        mock_rng.push_bool(false);
        mock_rng.push_bool(false);
        mock_rng.push_bool(false);
        mock_rng.push_bool(false);
        mock_rng.push_bool(false);

        match new_node_setting {
            NoChange => {
                mock_rng.push_bool(false);
            }
            NewInno => {
                mock_rng.push_bool(true);
                mock_rng.push_index(3, 4);

                expected_inno_index += 2;
                expected_node_index += 1;
                expected_new_innos += 2;
                expected_new_nodes += 1;
                expected_neuron_genes += 1;
                expected_synapse_genes += 2;
            }
            OldInno => {
                mock_rng.push_bool(true);
                mock_rng.push_index(2, 4);

                expected_neuron_genes += 1;
                expected_synapse_genes += 2;
            }
        }

        match new_edge_setting {
            NoChange => {
                mock_rng.push_bool(false);
            },
            NewInno => {
                mock_rng.push_bool(true);
                mock_rng.push_index(3, expected_neuron_genes);
                mock_rng.push_index(2, expected_neuron_genes);
                mock_rng.push_range_float(0.5, -1.0, 1.0);
                expected_inno_index += 1;
                expected_new_innos += 1;
                expected_synapse_genes += 1;
            }
            OldInno => {
                mock_rng.push_bool(true);
                mock_rng.push_index(2, expected_neuron_genes);
                mock_rng.push_index(3, expected_neuron_genes);
                mock_rng.push_range_float(0.5, -1.0, 1.0);
                expected_synapse_genes += 1;
            }
        }

        let (neuron_genes, synapse_genes) = mutate(&mut mock_rng, &params, &mut next_inno_index, &mut next_node_index, &mut new_innos, &mut new_nodes, neuron_genes, synapse_genes);

        assert_eq!(next_inno_index, expected_inno_index);
        assert_eq!(next_node_index, expected_node_index);
        assert_eq!(new_innos.len(), expected_new_innos);
        assert_eq!(new_nodes.len(), expected_new_nodes);
        assert_eq!(neuron_genes.len(), expected_neuron_genes);
        assert_eq!(synapse_genes.len(), expected_synapse_genes);

        match new_node_setting {
            NoChange => {}
            NewInno => {

                assert!(new_innos.contains_key(&(12, 15)));
                assert_eq!(*new_innos.get(&(12, 15)).unwrap(), 60);
                assert!(new_innos.contains_key(&(15, 4)));
                assert_eq!(*new_innos.get(&(15, 4)).unwrap(), 61);

                assert!(new_nodes.contains_key(&41));
                assert_eq!(*new_nodes.get(&41).unwrap(), 15);

                assert_eq!(neuron_genes[4].node_name, 15);

                assert_eq!(synapse_genes[5].inno_num, 60);
                assert_eq!(synapse_genes[6].inno_num, 61);
            }
            OldInno => {
                assert_eq!(neuron_genes[4].node_name, 14);

                assert_eq!(synapse_genes[5].inno_num, 57);
                assert_eq!(synapse_genes[6].inno_num, 58);
            }
        }

        match new_edge_setting {
            NoChange => {
            },
            NewInno => {
                assert!(new_innos.contains_key(&(12, 7)));
                assert_eq!(*new_innos.get(&(12, 7)).unwrap(), next_inno_index - 1);

                assert_eq!(synapse_genes[synapse_genes.len() - 1].inno_num, next_inno_index - 1);
            }
            OldInno => {
                assert_eq!(synapse_genes[synapse_genes.len() - 1].inno_num, 59);
            }
        }

    }
}