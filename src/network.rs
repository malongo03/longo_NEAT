use std::collections::HashMap;
use std::error::Error;
use std::mem::swap;
use crate::genome::{Genome, NeuronGene, NeuronType, SynapseGene};

pub enum Activation {
    Sigmoid,
    Relu,
    Tanh
}
impl Activation {
    #[inline]
    pub fn get_pointer(&self) -> fn(f64) -> f64 {
        match self {
            Activation::Sigmoid => |x| (1. + (-4.9 * x).exp()).recip(),
            Activation::Relu => todo!(),
            Activation::Tanh => todo!()
        }
    }
}

struct RnnEdge {
    src_id: usize,
    tgt_id: usize,
    weight: f64
}

pub struct RnnNetwork {
    node_state: Vec<f64>,
    new_node_state: Vec<f64>,
    edges: Vec<RnnEdge>,
    activation_function: fn(f64) -> f64,
    pub input_mapping: Vec<(usize, usize)>,
    pub output_mapping: Vec<(usize, usize)>,
}
impl RnnNetwork {
    fn new(activation_function: fn(f64) -> f64) -> Self {
        Self {
            node_state: vec![],
            new_node_state: vec![],
            edges: vec![],
            activation_function,
            input_mapping: vec![],
            output_mapping: vec![],
        }
    }

    fn new_from_genome(genome: &Genome, activation_function: fn(f64) -> f64) -> Self {
        let n: usize = genome.neuron_genes().len();
        let node_state: Vec<f64> = vec![0.0; n];
        let new_node_state: Vec<f64> = vec![0.0; n];

        let mut name_to_index: HashMap<usize, usize> = HashMap::with_capacity(n);
        let mut input_mapping: Vec<(usize, usize)> = Vec::new();
        let mut output_mapping: Vec<(usize, usize)> = Vec::new();
        for (node_id, node) in genome.neuron_genes().iter().enumerate() {
            name_to_index.insert(node.node_name, node_id);
            match node.neuron_type {
                NeuronType::Sensory(input_id) => {
                    input_mapping.push((input_id, node_id));
                }
                NeuronType::Muscular(output_id) => {
                    output_mapping.push((output_id, node_id));
                }
                _ => {}
            }
        }

        let mut edges: Vec<RnnEdge> = Vec::with_capacity(genome.genome_size());
        for edge in genome.synapse_genes() {
            if !edge.enabled {
                continue;
            }
            let src_id = name_to_index[&edge.src_name];
            let tgt_id = name_to_index[&edge.tgt_name];

            let rnn_edge = RnnEdge {
                src_id,
                tgt_id,
                weight: edge.weight,
            };

            edges.push(rnn_edge);
        }

        Self {
            node_state,
            new_node_state,
            edges,
            activation_function,
            input_mapping,
            output_mapping,
        }
    }

    fn tick(&mut self, inputs: &[f64], outputs: &mut [f64]) {
        // Pull inputs
        for (input_id, node_id) in self.input_mapping.iter() {
            self.node_state[*node_id] = inputs[*input_id];
        }
        // Use activation functions
        for node in self.node_state.iter_mut() {
            *node = (self.activation_function)(*node);
        }
        // Push outputs (we do this first because the end of this loop does not compute activation functions
        for (output_id, node_id) in self.output_mapping.iter() {
            outputs[*output_id] += self.node_state[*node_id];
        }

        // Compute next tick
        self.new_node_state.fill(0.0);
        for edge in self.edges.iter() {
            self.new_node_state[edge.tgt_id] += edge.weight * self.node_state[edge.src_id];
        }
        swap(&mut self.node_state, &mut self.new_node_state);
    }

    /// Note! This function has no Genome safety checks. A Network modified by it has no guarantee
    /// that it produces a valid Genome!
    fn add_edge(&mut self, src_id: usize, tgt_id: usize, weight: f64) {
        self.edges.push(RnnEdge{src_id, tgt_id, weight});
    }

    /// Note! This function has no Genome safety checks. A Network modified by it has no guarantee
    /// that it produces a valid Genome!
    fn add_or_replace_edge(&mut self, src_id: usize, tgt_id: usize, weight: f64) {
        for edge in self.edges.iter_mut() {
            if edge.src_id == src_id && edge.tgt_id == tgt_id {
                edge.weight = weight;
                return;
            }
        }
        self.edges.push(RnnEdge{src_id, tgt_id, weight});
    }

    /// Note! This function has no Genome safety checks. A Network modified by it has no guarantee
    /// that it produces a valid Genome!
    fn add_node(&mut self, input_id: Option<usize>, output_id: Option<usize>) {
        let node_id = self.node_state.len();
        self.node_state.push(0.0);
        self.new_node_state.push(0.0);

        if let Some(input_id) = input_id {
            self.input_mapping.push((input_id, node_id));
        }
        if let Some(output_id) = output_id {
            self.output_mapping.push((output_id, node_id));
        }
    }

    fn to_genome(&self) -> Result<Genome, Box<dyn Error>> {
        let n = self.node_state.len();

        let mut neuron_genes: Vec<NeuronGene> = Vec::with_capacity(n);
        for i in 0..n {
            neuron_genes.push(NeuronGene{node_name: i, neuron_type: NeuronType::Inter()})
        }
        for (input_id, node_id) in self.input_mapping.iter() {
            if let NeuronType::Sensory(_) = neuron_genes[*node_id].neuron_type {
                return Err("Genomes should not have Sensory neurons with multiple input ids".into())
            }
            neuron_genes[*node_id].neuron_type = NeuronType::Sensory(*input_id);
        }
        for (output_id, node_id) in self.output_mapping.iter() {
            match neuron_genes[*node_id].neuron_type {
                NeuronType::Muscular(_) => {
                    return Err("Genomes should not have Muscular neurons with multiple output ids".into())
                }
                NeuronType::Sensory(_) => {
                    return Err("Genome neurons should not be both Sensory and Muscular".into())
                }
                NeuronType::Inter() => {
                    neuron_genes[*node_id].neuron_type = NeuronType::Muscular(*output_id);
                }
            }
        }

        let m: usize = self.edges.len();
        let mut synapse_genes: Vec<SynapseGene> = Vec::with_capacity(m);
        for (i, edge) in self.edges.iter().enumerate() {
            synapse_genes.push(SynapseGene{
                src_name: edge.src_id,
                tgt_name: edge.tgt_id,
                weight: edge.weight,
                inno_num: i,
                enabled: true,
            })
        }

        Genome::new(0, 0, neuron_genes, synapse_genes)
    }
}