pub trait Simulation {
    fn tick(&mut self, actions: &[f64]) -> &[f64];

    fn num_inputs(&self) -> usize;
    fn num_outputs(&self) -> usize;
}