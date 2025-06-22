use ndarray::{Array, Array2, Array4, Dimension};

use crate::{layers::{CompositeLayer, RawLayer}, loss_functions::LossFunction, optimizers::Optimizer};

pub enum ThreadMode {
    Single,
    Multi(usize)
}

/// Expect examples in the format (data, labels)
pub fn train_model<M, D, O, L>(
    model: M,
    optimizer: O,
    loss_fn: L,
    examples: (Array<D, f32>, Array2<f32>),
    thread_mode: ThreadMode,
) where 
    M: CompositeLayer,
    D: Dimension,
    O: Optimizer,
    L: LossFunction
{
    // TODO: Rewrite below to be correct

    for (i, (x, labels)) in reshaped_train.axis_iter(Axis(0)).zip(reshaped_labels.axis_iter(Axis(0))).enumerate() {
        let batch_time = std::time::Instant::now();

        let mut label_encoded = Array2::<f32>::zeros((batch_size, num_classes));
        for (i, &label) in labels.iter().enumerate() {
            label_encoded[[i, label as usize]] = 1.;
        }

        // Go from (batch, 28, 28) to (batch, 1, 28, 28)
        let expanded = x.insert_axis(Axis(1)).map(|&v| v as f32 / 255.);

        let pred = network.forward(&expanded, true);
        let cost = CrossEntropyWithLogitsLoss::original(&pred.clone(), &label_encoded.clone());
        avg_cost += cost;

        // Compute accuracy for this batch
        let mut batch_acc = 0.;
        for (&label, preds) in labels.iter().zip(pred.axis_iter(Axis(0))) {
            if preds.argmax().unwrap() == label as usize {
                batch_acc += 1.;
            }
        }
        batch_acc /= batch_size as f32;
        avg_acc += batch_acc;

        println!("Batch {i:>3} | avg loss: {cost:>7.6} | avg acc: {:>6.2}% | time: {:.0}ms", batch_acc * 100., batch_time.elapsed().as_millis());

        // Back propagation
        let back = CrossEntropyWithLogitsLoss::derivative(&pred, &label_encoded);
        network.backward(&back);

        // Gradient application
        optimizer.step(&mut network.get_learnable_parameters());

        // Zero gradients before next epoch
        optimizer.zero_gradients(&mut network.get_learnable_parameters());
    }
}