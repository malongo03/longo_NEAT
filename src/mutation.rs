use rand::distr::Distribution;
use std::collections::HashMap;
use rand::distr::Uniform;
use rand::prelude::*;
use crate::genome::{NeuronGene, NeuronType, SynapseGene};

const MUTATE_GENE_WEIGHT: f64 = 0.8;
const MUTATE_RANDOM_WEIGHT: f64 = 0.1;
const MUTATE_ENABLE_PROB: f64 = 0.05;
const MUTATE_DISABLE_PROB: f64 = 0.005;
const MUTATION_POWER: f64 = 2.5;
const MUTATE_NEW_EDGE: f64 = 0.05;
const MUTATE_NEW_NODE: f64 = 0.03;

const CAP_WEIGHT: f64 = 8.0;
const CAP_NEW_WEIGHT: f64 = 1.0;

pub fn mutate<R: Rng>(rng: &mut R,
                      next_inno_index: &mut usize,
                      next_node_index: &mut usize,
                      mut neuron_genes: Vec<NeuronGene>,
                      mut synapse_genes: Vec<SynapseGene>,
                      new_innos: &mut HashMap<(usize, usize), usize>,
                      new_nodes: &mut HashMap<usize, usize>)
    -> (Vec<NeuronGene>, Vec<SynapseGene>) {
    let weight_dist = Uniform::new_inclusive(-MUTATION_POWER, MUTATION_POWER)
        .expect("Mutation power should be strictly positive and finite.");

    // Mutate weights on individual synapse genes.
    let num_genes = synapse_genes.len() as f64;
    for gene in synapse_genes.iter_mut() {
        mutate_edge(rng, weight_dist, num_genes, gene);
    }

    // Mutate a new edge.
    // Note that this is done with rejection sampling, and so it is not guaranteed to always
    // succeed. There are more clever ways to do this if synapse_genes was better organized, but
    // that defeats NEAT's homology requirements, which leaves us stuck with the lazy solution
    // unless an *extra* clever solution is found.
    if rng.random_bool(MUTATE_NEW_EDGE) {
        mutate_new_edge(rng, next_inno_index, &mut neuron_genes, &mut synapse_genes, new_innos);
    }

    // Mutate a new node.
    // This one is guaranteed to succeed.
    if rng.random_bool(MUTATE_NEW_NODE) {
        mutate_new_node(rng, next_inno_index, next_node_index, &mut neuron_genes, &mut synapse_genes, new_innos, new_nodes);
    }

    (neuron_genes, synapse_genes)
}

#[inline]
fn mutate_edge<R: Rng>(rng: &mut R,
                       weight_dist: Uniform<f64>,
                       num_genes: f64,
                       gene: &mut SynapseGene) {
    if rng.random_bool(MUTATE_GENE_WEIGHT) {
        let nudge = weight_dist.sample(rng);
        if rng.random_bool(MUTATE_RANDOM_WEIGHT) {
            gene.weight = nudge;
        } else {
            gene.weight += nudge;
            gene.weight = gene.weight.clamp(-CAP_WEIGHT, CAP_WEIGHT);
        }
    }

    if gene.enabled {
        if rng.random_bool(MUTATE_DISABLE_PROB / num_genes) {
            gene.enabled = false;
        }
    } else {
        if rng.random_bool(MUTATE_ENABLE_PROB / num_genes) {
            gene.enabled = true;
        }
    }
}

#[inline]
fn mutate_new_edge<R: Rng>(rng: &mut R,
                           next_inno_index: &mut usize,
                           neuron_genes: &Vec<NeuronGene>,
                           synapse_genes: &mut Vec<SynapseGene>,
                           new_innos: &mut HashMap<(usize, usize), usize>) {
    for _ in 0..20 {
        let src_id: usize = rng.random_range(0..neuron_genes.len());
        let tgt_id: usize = rng.random_range(0..neuron_genes.len());

        // Reject if the target is a sensory type neuron.
        if matches!(neuron_genes[tgt_id].neuron_type, NeuronType::Sensory(_)) {
            continue;
        }
        // TODO:
        // If we knew the number of sensory neurons already present, we can guarantee we do not
        // choose a sensory neuron by tgt_id displacement

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
                let new_num = *next_inno_index;
                *next_inno_index += 1;
                new_num
            });

        let weight: f64 = rng.random_range(-CAP_NEW_WEIGHT..=CAP_NEW_WEIGHT);
        let new_synapse: SynapseGene = SynapseGene {
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

#[inline]
fn mutate_new_node<R: Rng>(rng: &mut R,
                           next_inno_index: &mut usize,
                           next_node_index: &mut usize,
                           neuron_genes: &mut Vec<NeuronGene>,
                           synapse_genes: &mut Vec<SynapseGene>,
                           new_innos: &mut HashMap<(usize, usize), usize>,
                           new_nodes: &mut HashMap<usize, usize>) {

    // Choose enabled gene to split
    let target_idx = synapse_genes.iter()
        .enumerate()
        .filter(|(_, gene)| gene.enabled)
        .map(|(idx, _)| idx)
        .choose(rng);

    if let Some(target_idx) = target_idx {
        let old_synapse = synapse_genes[target_idx].clone();
        synapse_genes[target_idx].enabled = false;

        let new_node_id = *new_nodes.entry(old_synapse.inno_num).or_insert_with(|| {
            let id = *next_node_index;
            *next_node_index += 1;
            id
        });

        let is_new_split = new_node_id == *next_node_index - 1;

        let (inno1, inno2) = if is_new_split {
            let i1 = *next_inno_index;
            let i2 = *next_inno_index + 1;
            *next_inno_index += 2;

            new_innos.insert((old_synapse.src_id, new_node_id), i1);
            new_innos.insert((new_node_id, old_synapse.tgt_id), i2);

            (i1, i2)
        } else {
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

        neuron_genes.push(NeuronGene {
            node_name: new_node_id,
            neuron_type: NeuronType::Inter()
        })
    }
}


#[cfg(test)]
mod test_mutation {
    use super::*;

    ///////////////////////
    // MUTATE_EDGE TESTS //
    ///////////////////////

    #[test]
    fn test_mutate_edge_correct_probabilities() {todo!()}

    ///////////////////////////
    // MUTATE_NEW_EDGE TESTS //
    ///////////////////////////

    #[test]
    fn test_mutate_new_edge_new_inno() {todo!()}

    #[test]
    fn test_mutate_new_edge_repeat_inno() {todo!()}

    #[test]
    fn test_mutate_new_edge_no_sensory_targets() {todo!()}

    ///////////////////////////
    // MUTATE_NEW_NODE TESTS //
    ///////////////////////////

    #[test]
    fn test_mutate_new_node_new_inno() {todo!()}

    #[test]
    fn test_mutate_new_node_repeat_inno() {todo!()}

    #[test]
    fn test_mutate_new_node_correct_weights() {todo!()}

    #[test]
    fn test_mutate_new_node_all_genes_disabled() {todo!()}

    //////////////////
    // MUTATE TESTS //
    //////////////////

    #[test]
    fn test_mutation_can_mutate_edge() {todo!()}

    #[test]
    fn test_mutation_can_mutate_new_edge() {todo!()}

    #[test]
    fn test_mutation_can_mutate_new_node() {todo!()}

    #[test]
    fn test_mutation_correct_probabilities() {todo!()}
}