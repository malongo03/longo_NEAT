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
        .expect("Mutation power should be strictly positive and finite");

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
        .expect("Mutation power should be strictly positive and finite");

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
        if matches!(neuron_genes[tgt_id].neuron_type(), NeuronType::Sensory(_)) {
            continue;
        }
        // TODO:
        // If we knew the number of sensory neurons already present, we can guarantee we do not
        // choose a sensory neuron by tgt_id displacement. Perhaps worth counting earlier in the
        // mutation cycle.

        let src_name: usize = neuron_genes[src_id].node_name();
        let tgt_name: usize = neuron_genes[tgt_id].node_name();

        // Reject if this edge appears elsewhere.
        if synapse_genes.iter().any(
            |gene| gene.src_name() == src_name && gene.tgt_name() == tgt_name
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
        let new_synapse: SynapseGene = SynapseGene::new(
            src_name,
            tgt_name,
            weight,
            inno_num,
            true,
        );
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

        let is_new_split = !new_nodes.contains_key(&old_synapse.inno_num());

        let new_node_id = *new_nodes.entry(old_synapse.inno_num()).or_insert_with(|| {
            let id = *next_node_index;
            *next_node_index += 1;
            id
        });

        let (inno1, inno2) = if is_new_split {
            let i1 = *next_inno_index;
            let i2 = *next_inno_index + 1;
            *next_inno_index += 2;

            new_innos.insert((old_synapse.src_name(), new_node_id), i1);
            new_innos.insert((new_node_id, old_synapse.tgt_name()), i2);

            (i1, i2)
        } else {
            let i1 = *new_innos.get(&(old_synapse.src_name(), new_node_id))
                .expect("New node innovation already evolved");
            let i2 = i1 + 1;
            (i1, i2)
        };

        synapse_genes.push(SynapseGene::new(
            old_synapse.src_name(),
            new_node_id,
            1.0,
            inno1,
            old_synapse.enabled,
        ));

        synapse_genes.push(SynapseGene::new(
            new_node_id,
            old_synapse.tgt_name(),
            old_synapse.weight,
            inno2,
            true,
        ));

        neuron_genes.push(NeuronGene::new(new_node_id, NeuronType::Inter()));
    }
}


#[cfg(test)]
mod tests {
    // TODO: Consolidate initialization functions.
    use super::*;
    use crate::mock_rng::MockRng;
    use helpers::*;

    mod helpers {
        use super::*;

        pub fn produce_example_genes() -> (Vec<NeuronGene>, Vec<SynapseGene>) {
            let neuron_genes: Vec<NeuronGene> = vec![NeuronGene::new(0, NeuronType::Sensory(0)),
                                                     NeuronGene::new(4, NeuronType::Muscular(0)),
                                                     NeuronGene::new(7, NeuronType::Inter()),
                                                     NeuronGene::new(12, NeuronType::Inter())];

            let synapse_genes: Vec<SynapseGene> = vec![
                SynapseGene::new(0, 4, 1.0, 0, false),
                SynapseGene::new(0, 7, 1.0, 8, true),
                SynapseGene::new(7, 4, 1.0, 9, false),
                SynapseGene::new(0, 12, 2.0, 40, true),
                SynapseGene::new(12, 4, 2.0, 41, true)
            ];
            (neuron_genes, synapse_genes)
        }

        pub fn populate_rejections(mock_rng: &mut MockRng) {
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

    }

    mod mutate_edge_weight {
        use super::*;
        use test_case::test_case;

        #[test_case(7.5, 1.0, 8.0; "overflow positive 1")]
        #[test_case(6.5, 2.0, 8.0; "overflow positive 2")]
        #[test_case(5.0, 2.0, 7.0; "normal positive 1")]
        #[test_case(6.0, 1.0, 7.0; "normal positive 2")]
        #[test_case(-7.5, -1.0, -8.0; "underflow negative 1")]
        #[test_case(-6.5, -2.0, -8.0; "underflow negative 2")]
        #[test_case(-5.0, -2.0, -7.0; "normal negative 1")]
        #[test_case(-6.0, -1.0, -7.0; "normal negative 2")]
        fn adds_nudge(existing_weight: f64, nudge: f64, final_weight: f64) {
            let mut params = MutationParameters::default_params();
            params.mutation_power = 2.5;
            params.edge_weight_cap = 8.0;

            let distribution = Uniform::new_inclusive(-2.5, 2.5).unwrap();

            let mut mock_rng = MockRng::new();
            mock_rng.push_uniform_float_inclusive(nudge, -2.5, 2.5);
            mock_rng.push_bool(false);

            let mut gene = SynapseGene::new(
                0,
                1,
                existing_weight,
                0,
                false,
            );

            mutate_edge_weight(&mut mock_rng, &params, &distribution, &mut gene);

            assert!((gene.weight - final_weight).abs() <= f64::EPSILON,
                    "mutate_edge_weight add path failed to return the expected weight (likely a failed clamp)");
        }

        #[test_case(-7.5, 1.0; "1")]
        #[test_case(6.5, 2.0; "2")]
        #[test_case(-5.0, 2.0; "3")]
        #[test_case(6.0, 0.1; "4")]
        fn reassigns_nudge(existing_weight: f64, nudge: f64) {
            let mut params = MutationParameters::default_params();
            params.mutation_power = 2.5; // 2.5
            params.edge_weight_cap = 8.0; // 8.0

            let distribution = Uniform::new_inclusive(-2.5, 2.5).unwrap();

            let mut mock_rng = MockRng::new();
            let expected = mock_rng.push_uniform_float_inclusive(nudge, -2.5, 2.5);
            mock_rng.push_bool(true);

            let mut gene = SynapseGene::new(
                0,
                1,
                existing_weight,
                0,
                false
            );

            mutate_edge_weight(&mut mock_rng, &params, &distribution, &mut gene);

            assert_eq!(gene.weight, expected,
                       "mutate_edge_weight reassign mode failed to return the expected weight");
        }
    }

    mod mutate_new_edge {
        use super::*;

        #[test]
        fn correct_synapse_new_inno() {
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

            mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &neuron_genes, &mut synapse_genes);

            assert_eq!(synapse_genes.len(), 6,
                       "mutate_new_edge did not add the mutated synapse to the output on the new \
                       inno path");
            let expected_synapse = SynapseGene::new(
                12,
                7,
                weight,
                60,
                true,
            );
            let produced_synapse = &synapse_genes[5];
            assert_eq!(*produced_synapse, expected_synapse,
                       "mutate_new_edge did not produce the expected mutated synapse on the new \
                       inno path");
        }

        #[test]
        fn correct_synapse_new_inno_rejections() {
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

            mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &neuron_genes, &mut synapse_genes);

            let expected_synapse = SynapseGene::new(
                12,
                7,
                weight,
                60,
                true
            );
            let produced_synapse = &synapse_genes[5];

            assert_eq!(synapse_genes.len(), 6,
                       "mutate_new_edge did not add the mutated synapse to the output after prior \
                       rejections on the new inno path");
            assert_eq!(*produced_synapse, expected_synapse,
                       "mutate_new_edge did not produce the expected mutated synapse after prior \
                       rejections on the new inno path")
        }

        #[test]
        fn correct_synapse_repeat_inno() {
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

            mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &neuron_genes, &mut synapse_genes);

            assert_eq!(synapse_genes.len(), 6,
                       "mutate_new_edge did not add the mutated synapse to the output on the repeat \
                       inno path");
            let expected_synapse = SynapseGene::new(
                7,
                12,
                weight,
                59,
                true
            );
            let produced_synapse = &synapse_genes[5];
            assert_eq!(*produced_synapse, expected_synapse,
                       "mutate_new_edge did not produce the expected mutated synapse on the repeat \
                       inno path");
        }

        #[test]
        fn correct_synapse_repeat_inno_rejections() {
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

            mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &neuron_genes, &mut synapse_genes);

            let expected_synapse = SynapseGene::new(
                7,
                12,
                weight,
                59,
                true
            );
            let produced_synapse = &synapse_genes[5];

            assert_eq!(synapse_genes.len(), 6,
                       "mutate_new_edge did not add the mutated synapse to the output after prior \
                       rejections on the repeat inno path");
            assert_eq!(*produced_synapse, expected_synapse,
                       "mutate_new_edge did not produce the expected mutated synapse after prior \
                       rejections on the repeat inno path");
        }

        #[test]
        fn existing_synapses_static_new_inno() {
            let mut params = MutationParameters::default_params();
            params.edge_new_weight_cap = 1.0;

            let mut next_inno_index: usize = 60;

            let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
            new_innos.insert((7, 12), 59);

            let (neuron_genes, mut synapse_genes) = produce_example_genes();
            let old_synapse_genes = synapse_genes.clone();

            let mut mock_rng = MockRng::new();
            mock_rng.push_index(3, 4);
            mock_rng.push_index(2, 4);
            mock_rng.push_range_float(0.5, -1.0, 1.0);

            mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &neuron_genes, &mut synapse_genes);

            assert_eq!(synapse_genes[0], old_synapse_genes[0],
                       "mutate_new_edge mutated an existing synapse on the new inno path");
            assert_eq!(synapse_genes[1], old_synapse_genes[1],
                       "mutate_new_edge mutated an existing synapse on the new inno path");
            assert_eq!(synapse_genes[2], old_synapse_genes[2],
                       "mutate_new_edge mutated an existing synapse on the new inno path");
            assert_eq!(synapse_genes[3], old_synapse_genes[3],
                       "mutate_new_edge mutated an existing synapse on the new inno path");
            assert_eq!(synapse_genes[4], old_synapse_genes[4],
                       "mutate_new_edge mutated an existing synapse on the new inno path")
        }

        #[test]
        fn existing_synapses_static_new_inno_rejections() {
            let mut params = MutationParameters::default_params();
            params.edge_new_weight_cap = 1.0;

            let mut next_inno_index: usize = 60;

            let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
            new_innos.insert((7, 12), 59);

            let (neuron_genes, mut synapse_genes) = produce_example_genes();
            let old_synapse_genes = synapse_genes.clone();

            let mut mock_rng = MockRng::new();
            populate_rejections(&mut mock_rng);
            mock_rng.push_index(3, 4);
            mock_rng.push_index(2, 4);
            mock_rng.push_range_float(0.5, -1.0, 1.0);

            mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &neuron_genes, &mut synapse_genes);

            assert_eq!(synapse_genes[0], old_synapse_genes[0],
                       "mutate_new_edge mutated an existing synapse on the new inno path after prior rejections");
            assert_eq!(synapse_genes[1], old_synapse_genes[1],
                       "mutate_new_edge mutated an existing synapse on the new inno path after prior rejections");
            assert_eq!(synapse_genes[2], old_synapse_genes[2],
                       "mutate_new_edge mutated an existing synapse on the new inno path after prior rejections");
            assert_eq!(synapse_genes[3], old_synapse_genes[3],
                       "mutate_new_edge mutated an existing synapse on the new inno path after prior rejections");
            assert_eq!(synapse_genes[4], old_synapse_genes[4],
                       "mutate_new_edge mutated an existing synapse on the new inno path after prior rejections")
        }

        #[test]
        fn existing_synapses_static_repeat_inno() {
            let mut params = MutationParameters::default_params();
            params.edge_new_weight_cap = 1.0;

            let mut next_inno_index: usize = 60;

            let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
            new_innos.insert((7, 12), 59);

            let (neuron_genes, mut synapse_genes) = produce_example_genes();
            let old_synapse_genes = synapse_genes.clone();

            let mut mock_rng = MockRng::new();
            mock_rng.push_index(2, 4);
            mock_rng.push_index(3, 4);
            mock_rng.push_range_float(0.5, -1.0, 1.0);

            mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &neuron_genes, &mut synapse_genes);

            assert_eq!(synapse_genes[0], old_synapse_genes[0],
                       "mutate_new_edge mutated an existing synapse on the repeat inno path");
            assert_eq!(synapse_genes[1], old_synapse_genes[1],
                       "mutate_new_edge mutated an existing synapse on the repeat inno path");
            assert_eq!(synapse_genes[2], old_synapse_genes[2],
                       "mutate_new_edge mutated an existing synapse on the repeat inno path");
            assert_eq!(synapse_genes[3], old_synapse_genes[3],
                       "mutate_new_edge mutated an existing synapse on the repeat inno path");
            assert_eq!(synapse_genes[4], old_synapse_genes[4],
                       "mutate_new_edge mutated an existing synapse on the repeat inno path")
        }

        #[test]
        fn existing_synapses_static_repeat_inno_rejections() {
            let mut params = MutationParameters::default_params();
            params.edge_new_weight_cap = 1.0;

            let mut next_inno_index: usize = 60;

            let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
            new_innos.insert((7, 12), 59);

            let (neuron_genes, mut synapse_genes) = produce_example_genes();
            let old_synapse_genes = synapse_genes.clone();

            let mut mock_rng = MockRng::new();
            populate_rejections(&mut mock_rng);
            mock_rng.push_index(2, 4);
            mock_rng.push_index(3, 4);
            mock_rng.push_range_float(0.5, -1.0, 1.0);

            mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &neuron_genes, &mut synapse_genes);

            assert_eq!(synapse_genes[0], old_synapse_genes[0],
                       "mutate_new_edge mutated an existing synapse on the repeat inno path after prior rejections");
            assert_eq!(synapse_genes[1], old_synapse_genes[1],
                       "mutate_new_edge mutated an existing synapse on the repeat inno path after prior rejections");
            assert_eq!(synapse_genes[2], old_synapse_genes[2],
                       "mutate_new_edge mutated an existing synapse on the repeat inno path after prior rejections");
            assert_eq!(synapse_genes[3], old_synapse_genes[3],
                       "mutate_new_edge mutated an existing synapse on the repeat inno path after prior rejections");
            assert_eq!(synapse_genes[4], old_synapse_genes[4],
                       "mutate_new_edge mutated an existing synapse on the repeat inno path after prior rejections")
        }

        #[test]
        fn new_inno_updates_global_innovation() {
            let mut params = MutationParameters::default_params();
            params.edge_new_weight_cap = 1.0;

            let mut next_inno_index: usize = 60;

            let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
            new_innos.insert((7, 12), 59);

            let (neuron_genes, mut synapse_genes) = produce_example_genes();

            let mut mock_rng = MockRng::new();
            mock_rng.push_index(3, 4);
            mock_rng.push_index(2, 4);
            mock_rng.push_range_float(0.5, -1.0, 1.0);

            mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &neuron_genes, &mut synapse_genes);

            assert_eq!(next_inno_index, 61,
                       "mutate_new_edge did not update the global innovation number with a new \
                       innovation on the new inno path");
            assert_eq!(new_innos.len(), 2,
                       "mutate_new_edge did not update the innovation hash with a new innovation \
                       on the new inno path");
            assert!(new_innos.contains_key(&(12, 7)),
                    "mutate_new_edge did not add the correct innovation to the innovation hash \
                    on the new inno path");
            assert_eq!(*new_innos.get(&(12, 7)).unwrap(), 60,
                       "mutate_new_edge did not give the correct innovation number to the innovation \
                       hash on the new inno path");
            assert_eq!(*new_innos.get(&(7, 12)).unwrap(), 59,
                       "mutate_new_edge overwrote the innovation number of existing innovations \
                       on the new inno path");
        }

        #[test]
        fn new_inno_updates_global_innovation_rejections() {
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
            mock_rng.push_range_float(0.5, -1.0, 1.0);

            mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &neuron_genes, &mut synapse_genes);

            assert_eq!(next_inno_index, 61,
                       "mutate_new_edge did not update the global innovation number with a new \
                       innovation after prior rejections on the new inno path");
            assert_eq!(new_innos.len(), 2,
                       "mutate_new_edge did not update the innovation hash with a new innovation \
                       after prior rejections on the new inno path");
            assert!(new_innos.contains_key(&(12, 7)),
                    "mutate_new_edge did not add the correct innovation to the innovation hash \
                    after prior rejections on the new inno path");
            assert_eq!(*new_innos.get(&(12, 7)).unwrap(), 60,
                       "mutate_new_edge did not give the correct innovation number to the innovation \
                       hash after prior rejections on the new inno path");
            assert_eq!(*new_innos.get(&(7, 12)).unwrap(), 59,
                       "mutate_new_edge overwrote the innovation number of existing innovations \
                       after prior rejections on the new inno path");
        }

        #[test]
        fn repeat_inno_static_global_innovation() {
            let mut params = MutationParameters::default_params();
            params.edge_new_weight_cap = 1.0;

            let mut next_inno_index: usize = 60;

            let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
            new_innos.insert((7, 12), 59);

            let (neuron_genes, mut synapse_genes) = produce_example_genes();

            let mut mock_rng = MockRng::new();
            mock_rng.push_index(2, 4);
            mock_rng.push_index(3, 4);
            mock_rng.push_range_float(0.5, -1.0, 1.0);

            mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &neuron_genes, &mut synapse_genes);

            assert_eq!(next_inno_index, 60,
                       "mutate_new_edge updated the global innovation number for a repeat \
                       innovation on the repeat inno path");
            assert_eq!(new_innos.len(), 1,
                       "mutate_new_edge updated the innovation hash for a repeat innovation on the \
                       repeat inno path");
            assert_eq!(*new_innos.get(&(7, 12)).unwrap(), 59,
                       "mutate_new_edge overwrote the innovation number of existing \
                       innovations on the repeat inno path")
        }

        #[test]
        fn repeat_inno_static_global_innovation_rejections() {
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
            mock_rng.push_range_float(0.5, -1.0, 1.0);

            mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &neuron_genes, &mut synapse_genes);

            assert_eq!(next_inno_index, 60,
                       "mutate_new_edge updated the global innovation number for a repeat \
                       innovation after rejections on the repeat inno path");
            assert_eq!(new_innos.len(), 1,
                       "mutate_new_edge updated the innovation hash for a repeat innovation after \
                       rejections on the repeat inno path");
            assert_eq!(*new_innos.get(&(7, 12)).unwrap(), 59,
                       "mutate_new_edge overwrote the innovation number of an existing innovation \
                       after rejections on the repeat innovation path");
        }

        #[test]
        fn no_sensory_targets() {
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

            mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &neuron_genes, &mut synapse_genes);

            assert!(!new_innos.contains_key(&(12, 0)),
                    "mutate_new_edge chose a sensory neuron as a target");

            let produced_synapse = &synapse_genes[5];

            assert_ne!(produced_synapse.tgt_name(), 0,
                       "mutate_new_edge chose a sensory neuron as a target");
        }

        #[test]
        fn no_repeat_edges() {
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

            mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &neuron_genes, &mut synapse_genes);

            assert!(!new_innos.contains_key(&(0, 4)),
                    "mutate_new_edge duplicated an existing edge");

            let produced_synapse = &synapse_genes[5];

            assert_ne!((produced_synapse.src_name(), produced_synapse.tgt_name()), (0, 4),
                       "mutate_new_edge duplicated an existing edge");
        }

        #[test]
        fn graceful_rejection_exit() {
            let mut params = MutationParameters::default_params();
            params.edge_new_weight_cap = 1.0;

            let mut next_inno_index: usize = 60;

            let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
            new_innos.insert((7, 12), 59);

            let (neuron_genes, mut synapse_genes) = produce_example_genes();
            let old_synapse_genes = synapse_genes.clone();

            let mut mock_rng = MockRng::new();
            populate_rejections(&mut mock_rng);
            mock_rng.push_index(3, 4);
            mock_rng.push_index(1, 4);
            mock_rng.push_range_float(0.5, -1.0, 1.0);

            mutate_new_edge(&mut mock_rng, &params, &mut next_inno_index, &mut new_innos, &neuron_genes, &mut synapse_genes);

            assert_eq!(synapse_genes, old_synapse_genes,
                       "mutate_new_edge mutated synapses after 20 rejections");

            assert_eq!(next_inno_index, 60,
                       "mutate_new_edge mutated innovation structures after failing to mutate a synapse");
            assert_eq!(new_innos.len(), 1,
                       "mutate_new_edge mutated the innovation hash after failing to mutate a synapse");
            assert_eq!(*new_innos.get(&(7, 12)).unwrap(), 59,
                       "mutate_new_edge mutated an existing innovation after failing to mutate a synapse");
        }
    }

    mod mutate_new_node {
        use super::*;

        #[test]
        fn correct_node_new_inno() {
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

            assert_eq!(neuron_genes.len(), 5,
                       "mutate_new_node failed to mutate a valid new node on the new inno path");
            let new_neuron = &neuron_genes[4];
            assert_eq!(new_neuron.node_name(), 15,
                       "mutate_new_node did not mutate the correct node on the new inno path");
            assert!(matches!(new_neuron.neuron_type(), NeuronType::Inter()),
                    "mutate_new_node mutated a new node with NeuronType that was not Inter \
                    (update this test if this was added intentionally)");
        }

        #[test]
        fn correct_node_repeat_inno() {
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
            mock_rng.push_index(2, 4);

            mutate_new_node(&mut mock_rng, &mut next_inno_index, &mut next_node_index, &mut new_innos, &mut new_nodes, &mut neuron_genes, &mut synapse_genes);

            assert_eq!(neuron_genes.len(), 5,
                       "mutate_new_node failed to mutate a valid new node on the repeat inno path");
            let new_neuron = &neuron_genes[4];
            assert_eq!(new_neuron.node_name(), 14,
                       "mutate_new_node did not mutate the correct node on the repeat inno path");
            assert!(matches!(new_neuron.neuron_type(), NeuronType::Inter()),
                    "mutate_new_node mutated a new node with NeuronType that was not Inter \
                    (update this test if this was added intentionally)");
        }

        #[test]
        fn correct_synapses_new_inno() {
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

            assert_eq!(synapse_genes.len(), 7,
                       "mutate_new_node failed to add synapses after node mutation on the new inno path");
            let expected_new_synapse = SynapseGene::new(
                12,
                15,
                1.0,
                60,
                true,
            );
            let actual_new_synapse = &synapse_genes[5];
            assert_eq!(*actual_new_synapse, expected_new_synapse,
                       "mutate_new_node did not mutate the correct synapses on the new inno path");

            let expected_new_synapse = SynapseGene::new(
                15,
                4,
                2.0,
                61,
                true,
            );
            let actual_new_synapse = &synapse_genes[6];
            assert_eq!(*actual_new_synapse, expected_new_synapse,
                       "mutate_new_node did not mutate the correct synapses on the new inno path");
        }

        #[test]
        fn correct_synapses_repeat_inno() {
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
            mock_rng.push_index(2, 4);

            mutate_new_node(&mut mock_rng, &mut next_inno_index, &mut next_node_index, &mut new_innos, &mut new_nodes, &mut neuron_genes, &mut synapse_genes);

            assert_eq!(synapse_genes.len(), 7,
                       "mutate_new_node failed to add synapses after node mutation on the repeat inno path");
            let expected_new_synapse = SynapseGene::new(
                0,
                14,
                1.0,
                57,
                true
            );
            let actual_new_synapse = &synapse_genes[5];
            assert_eq!(*actual_new_synapse, expected_new_synapse,
                       "mutate_new_node did not mutate the correct synapses on the repeat inno path");

            let expected_new_synapse = SynapseGene::new(
                14,
                12,
                2.0,
                58,
                true
            );
            let actual_new_synapse = &synapse_genes[6];
            assert_eq!(*actual_new_synapse, expected_new_synapse,
                       "mutate_new_node did not mutate the correct synapses on the repeat inno path");
        }

        #[test]
        fn existing_synapses_static_new_inno() {
            let mut next_inno_index: usize = 60;
            let mut next_node_index: usize = 15;

            let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
            new_innos.insert((0, 14), 57);
            new_innos.insert((14, 12), 58);
            new_innos.insert((7, 12), 59);

            let mut new_nodes: HashMap<usize, usize> = HashMap::new();
            new_nodes.insert(40, 14);

            let (mut neuron_genes, mut synapse_genes) = produce_example_genes();
            let old_synapse_genes = synapse_genes.clone();

            let mut mock_rng = MockRng::new();
            mock_rng.push_index(3, 4);

            mutate_new_node(&mut mock_rng, &mut next_inno_index, &mut next_node_index, &mut new_innos, &mut new_nodes, &mut neuron_genes, &mut synapse_genes);

            assert_eq!(synapse_genes[0], old_synapse_genes[0],
                       "mutate_new_node mutated an existing synapse that was not selected on the new inno path");
            assert_eq!(synapse_genes[1], old_synapse_genes[1],
                       "mutate_new_node mutated an existing synapse that was not selected on the new inno path");
            assert_eq!(synapse_genes[2], old_synapse_genes[2],
                       "mutate_new_node mutated an existing synapse that was not selected on the new inno path");
            assert_eq!(synapse_genes[3], old_synapse_genes[3],
                       "mutate_new_node mutated an existing synapse that was not selected on the new inno path");

            assert_eq!(synapse_genes[4].weight, old_synapse_genes[4].weight,
                       "mutate_new_node mutated the weight of the selected synapse on the new inno path");
            assert!(!synapse_genes[4].enabled,
                    "mutate_new_node did not disable the selected synapse on the new inno path");
        }

        #[test]
        fn existing_synapses_static_repeat_inno() {
            let mut next_inno_index: usize = 60;
            let mut next_node_index: usize = 15;

            let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
            new_innos.insert((0, 14), 57);
            new_innos.insert((14, 12), 58);
            new_innos.insert((7, 12), 59);

            let mut new_nodes: HashMap<usize, usize> = HashMap::new();
            new_nodes.insert(40, 14);

            let (mut neuron_genes, mut synapse_genes) = produce_example_genes();
            let old_synapse_genes = synapse_genes.clone();

            let mut mock_rng = MockRng::new();
            mock_rng.push_index(2, 4);

            mutate_new_node(&mut mock_rng, &mut next_inno_index, &mut next_node_index, &mut new_innos, &mut new_nodes, &mut neuron_genes, &mut synapse_genes);

            assert_eq!(synapse_genes[0], old_synapse_genes[0],
                       "mutate_new_node mutated an existing synapse that was not selected on the repeat inno path");
            assert_eq!(synapse_genes[1], old_synapse_genes[1],
                       "mutate_new_node mutated an existing synapse that was not selected on the repeat inno path");
            assert_eq!(synapse_genes[2], old_synapse_genes[2],
                       "mutate_new_node mutated an existing synapse that was not selected on the repeat inno path");
            assert_eq!(synapse_genes[4], old_synapse_genes[4],
                       "mutate_new_node mutated an existing synapse that was not selected on the repeat inno path");

            assert_eq!(synapse_genes[3].weight, old_synapse_genes[3].weight,
                       "mutate_new_node mutated the weight of the selected synapse on the repeat inno path");
            assert!(!synapse_genes[3].enabled,
                    "mutate_new_node did not disable the selected synapse on the repeat inno path");
        }

        #[test]
        fn new_inno_updates_global_node_name() {
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

            assert_eq!(next_node_index, 16,
                       "mutate_new_node did not increment next_node_index on the new inno path");
            assert_eq!(new_nodes.len(), 2,
                       "mutate_new_node did not add a novel mutated node to the new_nodes hash");
            assert!(new_nodes.contains_key(&41),
                    "mutate_new_node did not add the correct novel mutated node to the new_nodes hash");
            assert_eq!(*new_nodes.get(&41).unwrap(), 15,
                       "mutate_new_node did not assign the correct inno_num for the novel mutated \
                       node on the new_nodes hash");
            assert_eq!(*new_nodes.get(&40).unwrap(), 14,
                       "mutate_new_node altered the inno num of an existing node in the new_nodes hash \
                       on the new inno path")
        }

        #[test]
        fn repeat_inno_static_global_node_name() {
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
            mock_rng.push_index(2, 4);

            mutate_new_node(&mut mock_rng, &mut next_inno_index, &mut next_node_index, &mut new_innos, &mut new_nodes, &mut neuron_genes, &mut synapse_genes);

            assert_eq!(next_node_index, 15,
                       "mutate_new_node mutated the next_node_index on the repeat inno path");
            assert_eq!(new_nodes.len(), 1,
                       "mutate_new_node added to the new_nodes hash on the repeat inno path");
            assert_eq!(*new_nodes.get(&40).unwrap(), 14,
                       "mutate_new_node altered the inno num of an existing node in the new_nodes hash \
                       on the repeat inno path")
        }

        #[test]
        fn new_inno_updates_global_innovation() {
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

            assert_eq!(next_inno_index, 62,
                       "mutate_new_node did not increment the next_inno_index on the new inno path");
            assert_eq!(new_innos.len(), 5,
                       "mutate_new_node did not add a novel innovations to the new_innos hash \
                       on the new inno path");
            assert!(new_innos.contains_key(&(12, 15)),
                    "mutate_new_node did not add the correct innovation to the new_innos hash \
                    on the new inno path");
            assert_eq!(*new_innos.get(&(12, 15)).unwrap(), 60,
                       "mutate_new_node did not assign the correct innovation numbers to the novel \
                       innovation on the new innos path");
            assert!(new_innos.contains_key(&(15, 4)),
                    "mutate_new_node did not add the correct innovation to the new_innos hash");
            assert_eq!(*new_innos.get(&(15, 4)).unwrap(), 61,
                       "mutate_new_node did not assign the correct innovation numbers to the novel \
                       innovation on the new innos path");
            assert_eq!(*new_innos.get(&(0, 14)).unwrap(), 57,
                       "mutate_new_node altered the inno num of an existing innovation on the new_innos hash");
            assert_eq!(*new_innos.get(&(14, 12)).unwrap(), 58,
                       "mutate_new_node altered the inno num of an existing innovation on the new_innos hash");
            assert_eq!(*new_innos.get(&(7, 12)).unwrap(), 59,
                       "mutate_new_node altered the inno num of an existing innovation on the new_innos hash");
        }

        #[test]
        fn repeat_inno_static_global_innovation() {
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
            mock_rng.push_index(2, 4);

            mutate_new_node(&mut mock_rng, &mut next_inno_index, &mut next_node_index, &mut new_innos, &mut new_nodes, &mut neuron_genes, &mut synapse_genes);

            assert_eq!(next_inno_index, 60,
                       "mutate_new_node mutated the next_inno_index on the repeat inno path");

            assert_eq!(new_innos.len(), 3,
                       "mutate_new_node added to the new_innos hash on the repeat inno path");
            assert_eq!(*new_innos.get(&(0, 14)).unwrap(), 57,
                       "mutate_new_node altered the inno num of an existing innovation on the new_innos hash");
            assert_eq!(*new_innos.get(&(14, 12)).unwrap(), 58,
                       "mutate_new_node altered the inno num of an existing innovation on the new_innos hash");
            assert_eq!(*new_innos.get(&(7, 12)).unwrap(), 59,
                       "mutate_new_node altered the inno num of an existing innovation on the new_innos hash");
        }

        #[test]
        fn test_mutate_new_node_all_genes_disabled() {
            let mut next_inno_index: usize = 60;
            let mut next_node_index: usize = 15;

            let mut new_innos: HashMap<(usize, usize), usize> = HashMap::new();
            new_innos.insert((0, 14), 57);
            new_innos.insert((14, 12), 58);
            new_innos.insert((7, 12), 59);
            let old_inno_hash = new_innos.clone();

            let mut new_nodes: HashMap<usize, usize> = HashMap::new();
            new_nodes.insert(40, 14);
            let old_node_hash = new_nodes.clone();

            let (mut neuron_genes, mut synapse_genes) = produce_example_genes();
            synapse_genes[0].enabled = false;
            synapse_genes[1].enabled = false;
            synapse_genes[2].enabled = false;
            synapse_genes[3].enabled = false;
            synapse_genes[4].enabled = false;
            let old_neuron_genes = neuron_genes.clone();
            let old_synapse_genes = synapse_genes.clone();

            let mut mock_rng = MockRng::new();
            mock_rng.push_index(2, 4);

            mutate_new_node(&mut mock_rng, &mut next_inno_index, &mut next_node_index, &mut new_innos, &mut new_nodes, &mut neuron_genes, &mut synapse_genes);

            assert_eq!(old_neuron_genes, neuron_genes, "mutate_new_node should have no effects if all genes are disabled");
            assert_eq!(old_synapse_genes, synapse_genes, "mutate_new_node should have no effects if all genes are disabled");

            assert_eq!(next_inno_index, 60, "mutate_new_node should have no effects if all genes are disabled");
            assert_eq!(next_node_index, 15, "mutate_new_node should have no effects if all genes are disabled");

            assert_eq!(old_inno_hash, new_innos, "mutate_new_node should have no effects if all genes are disabled");
            assert_eq!(old_node_hash, new_nodes, "mutate_new_node should have no effects if all genes are disabled");
        }
    }

    //////////////////
    // MUTATE TESTS //
    //////////////////

    mod mutation {
        use super::*;
        use test_case::test_case;

        #[test]
        fn can_toggle_genes() {
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

            assert!(synapse_genes[0].enabled, "mutation failed to re-enable a disabled gene");
            assert!(!synapse_genes[1].enabled, "mutation failed to disable an enabled gene");
            assert!(!synapse_genes[2].enabled, "mutation toggled a disabled gene it was not allowed to");
            assert!(synapse_genes[3].enabled, "mutation toggled an enabled gene it was not allowed to");
            assert!(synapse_genes[4].enabled, "mutation toggle an enabled gene it was not allowed to");
        }

        #[test]
        fn can_mutate_weights() {
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
            // 0: weight change and toggle
            mock_rng.push_bool(true);
            let weight_1 = mock_rng.push_uniform_float_inclusive(1.1, -2.5, 2.5) + 1.0;
            mock_rng.push_bool(false);
            mock_rng.push_bool(true);

            // 1: weight change
            mock_rng.push_bool(true);
            let weight_2 = mock_rng.push_uniform_float_inclusive(2.3, -2.5, 2.5);
            mock_rng.push_bool(true);
            mock_rng.push_bool(false);

            // 2: toggle
            mock_rng.push_bool(false);
            mock_rng.push_bool(true);

            // 3: no change
            mock_rng.push_bool(false);
            mock_rng.push_bool(false);

            // 4: toggle
            mock_rng.push_bool(false);
            mock_rng.push_bool(true);

            // No new edges
            mock_rng.push_bool(false);

            // No new nodes
            mock_rng.push_bool(false);

            let (_, synapse_genes) = mutate(&mut mock_rng, &params, &mut next_inno_index, &mut next_node_index, &mut new_innos, &mut new_nodes, neuron_genes, synapse_genes);

            assert!(synapse_genes[0].enabled, "mutate did not toggle synapse 0 on");
            assert!(synapse_genes[1].enabled, "mutate toggled synapse 1 off");
            assert!(synapse_genes[2].enabled, "mutate did not toggle synapse 2 on");
            assert!(synapse_genes[3].enabled, "mutate toggled synapse 3 off");
            assert!(!synapse_genes[4].enabled, "mutate did not toggle synapse 4 off");

            assert_eq!(synapse_genes[0].weight, weight_1, "mutate did not mutate synapse 0 to the correct weight");
            assert_eq!(synapse_genes[1].weight, weight_2, "mutate did not mutate synapse 1 to the correct weight");
            assert_eq!(synapse_genes[2].weight, 1.0, "mutate mutated synapse 2's weight");
            assert_eq!(synapse_genes[3].weight, 2.0, "mutate mutated synapse 3's weight");
            assert_eq!(synapse_genes[4].weight, 2.0, "mutate mutated synapse 4's weight");
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
        fn can_mutate_new_edge_and_new_node(new_edge_setting: IntegrationSetting, new_node_setting: IntegrationSetting) {
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
            let old_synapse_genes = synapse_genes.clone();
            let mut expected_neuron_genes: usize = 4;
            let mut expected_synapse_genes: usize = 5;

            let mut mock_rng = MockRng::new();
            // No change to existing synapses by weight/enable mutations
            for _ in 0..5 {
                mock_rng.push_bool(false);
                mock_rng.push_bool(false);
            }

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

            let weight: Option<f64> = match new_edge_setting {
                NoChange => {
                    mock_rng.push_bool(false);
                    None
                },
                NewInno => {
                    mock_rng.push_bool(true);
                    mock_rng.push_index(3, expected_neuron_genes);
                    mock_rng.push_index(2, expected_neuron_genes);
                    let weight = mock_rng.push_range_float(0.5, -1.0, 1.0);
                    expected_inno_index += 1;
                    expected_new_innos += 1;
                    expected_synapse_genes += 1;
                    Some(weight)
                }
                OldInno => {
                    mock_rng.push_bool(true);
                    mock_rng.push_index(2, expected_neuron_genes);
                    mock_rng.push_index(3, expected_neuron_genes);
                    let weight = mock_rng.push_range_float(0.5, -1.0, 1.0);
                    expected_synapse_genes += 1;
                    Some(weight)
                }
            };

            let (neuron_genes, synapse_genes) = mutate(&mut mock_rng, &params, &mut next_inno_index, &mut next_node_index, &mut new_innos, &mut new_nodes, neuron_genes, synapse_genes);

            assert_eq!(next_inno_index, expected_inno_index,
                       "mutate did not produce the expected next_inno_index");
            assert_eq!(next_node_index, expected_node_index,
                       "mutate did not produce the expected next_node_index");
            assert_eq!(new_innos.len(), expected_new_innos,
                       "mutate did not produce the correct number of innovations");
            assert_eq!(new_nodes.len(), expected_new_nodes,
                       "mutate did not produce the correct number of evolved nodes");
            assert_eq!(neuron_genes.len(), expected_neuron_genes,
                       "mutate did not produce the correct number of neuron genes");
            assert_eq!(synapse_genes.len(), expected_synapse_genes,
                       "mutate did not produce the correct number of synapse genes");

            assert_eq!(*new_innos.get(&(0, 14)).unwrap(), 57,
                       "mutate should not overwrite the innovation number of existing innovations");
            assert_eq!(*new_innos.get(&(14, 12)).unwrap(), 58,
                       "mutate should not overwrite the innovation number of existing innovations");
            assert_eq!(*new_innos.get(&(7, 12)).unwrap(), 59,
                       "mutate should not overwrite the innovation number of existing innovations");

            assert_eq!(*new_nodes.get(&40).unwrap(), 14,
                       "mutate should not overwrite the node name of synapse's split evolution");

            assert_eq!(synapse_genes[0], old_synapse_genes[0],
                       "mutate_new_edge and mutate_new_node should not touch existing synapses unless they are split");
            assert_eq!(synapse_genes[1], old_synapse_genes[1],
                       "mutate_new_edge and mutate_new_node should not touch existing synapses unless they are split");
            assert_eq!(synapse_genes[2], old_synapse_genes[2],
                       "mutate_new_edge and mutate_new_node should not touch existing synapses unless they are split");

            match new_node_setting {
                NoChange => {
                    assert_eq!(synapse_genes[3], old_synapse_genes[3],
                               "mutate_new_edge and mutate_new_node should not touch existing synapses unless they are split");
                    assert_eq!(synapse_genes[4], old_synapse_genes[4],
                               "mutate_new_edge and mutate_new_node should not touch existing synapses unless they are split");
                }
                NewInno => {
                    assert_eq!(synapse_genes[3], old_synapse_genes[3],
                               "mutate_new_edge and mutate_new_node should not touch existing synapses unless they are split");
                    assert_eq!(synapse_genes[4].weight, old_synapse_genes[4].weight,
                               "mutate_new_node should not mutate a split synapse's weight");
                    assert!(!synapse_genes[4].enabled, "mutate_new_node did not disable its split synapse");

                    assert!(new_innos.contains_key(&(12, 15)),
                            "mutate_new_node did not mutate the correct synapses");
                    assert_eq!(*new_innos.get(&(12, 15)).unwrap(), 60,
                               "mutate_new_node did not assign its synapses the correct innovation number");
                    assert!(new_innos.contains_key(&(15, 4)),
                            "mutate_new_node did not mutate the correct synapses");
                    assert_eq!(*new_innos.get(&(15, 4)).unwrap(), 61,
                               "mutate_new_node did not assign its synapses the correct innovation number");

                    assert!(new_nodes.contains_key(&41),
                            "mutate_new_node did not record the correct split synapse");
                    assert_eq!(*new_nodes.get(&41).unwrap(), 15,
                               "mutate_new_node did not assign its split synapse the correct node name");

                    assert_eq!(neuron_genes[4].node_name(), 15,
                               "mutate_new_node did not mutate the correct node");
                    assert!(matches!(neuron_genes[4].neuron_type(), NeuronType::Inter()),
                            "mutate_new_node did not mutate an inter node");

                    let expected_new_synapse = SynapseGene::new(
                        12,
                        15,
                        1.0,
                        60,
                        true,
                    );
                    let actual_new_synapse = &synapse_genes[5];
                    assert_eq!(*actual_new_synapse, expected_new_synapse,
                               "mutate_new_node did not mutate the correct synapses");

                    let expected_new_synapse = SynapseGene::new(
                        15,
                        4,
                        2.0,
                        61,
                        true,
                    );
                    let actual_new_synapse = &synapse_genes[6];
                    assert_eq!(*actual_new_synapse, expected_new_synapse,
                               "mutate_new_node did not mutate the correct synapses");
                }
                OldInno => {
                    assert_eq!(synapse_genes[4], old_synapse_genes[4],
                               "mutate_new_edge and mutate_new_node should not touch existing synapses unless they are split");
                    assert_eq!(synapse_genes[3].weight, old_synapse_genes[3].weight,
                               "mutate_new_node should not mutate a split synapse's weight");
                    assert!(!synapse_genes[3].enabled, "mutate_new_node did not disable its split synapse");

                    assert_eq!(neuron_genes[4].node_name(), 14,
                               "mutate_new_node did not mutate the correct node");
                    assert!(matches!(neuron_genes[4].neuron_type(), NeuronType::Inter()),
                            "mutate_new_node did not mutate an inter node");

                    let expected_new_synapse = SynapseGene::new(
                        0,
                        14,
                        1.0,
                        57,
                        true
                    );
                    let actual_new_synapse = &synapse_genes[5];
                    assert_eq!(*actual_new_synapse, expected_new_synapse,
                               "mutate_new_node did not mutate the correct synapses on the repeat inno path");

                    let expected_new_synapse = SynapseGene::new(
                        14,
                        12,
                        2.0,
                        58,
                        true
                    );
                    let actual_new_synapse = &synapse_genes[6];
                    assert_eq!(*actual_new_synapse, expected_new_synapse,
                               "mutate_new_node did not mutate the correct synapses on the repeat inno path");
                }
            }

            match new_edge_setting {
                NoChange => {},
                NewInno => {
                    assert!(new_innos.contains_key(&(12, 7)),
                            "mutate_new_edge did not record the correct new synapse");
                    assert_eq!(*new_innos.get(&(12, 7)).unwrap(), next_inno_index - 1,
                               "mutate_new_edge did not assign its synapse the correct innovation number");

                    let expected_synapse = SynapseGene::new(
                        12,
                        7,
                        weight.expect("Weight should have been assigned by this function"),
                        next_inno_index - 1,
                        true,
                    );
                    let produced_synapse = &synapse_genes[synapse_genes.len() - 1];
                    assert_eq!(*produced_synapse, expected_synapse,
                               "mutate_new_edge did not mutate the correct synapse");
                }
                OldInno => {
                    let expected_synapse = SynapseGene::new(
                        7,
                        12,
                        weight.expect("Weight should have been assigned by this function"),
                        59,
                        true
                    );
                    let produced_synapse = &synapse_genes[synapse_genes.len() - 1];
                    assert_eq!(*produced_synapse, expected_synapse,
                               "mutate_new_edge did not produce the expected mutated synapse on the repeat \
                       inno path");
                }
            }

        }
    }


}