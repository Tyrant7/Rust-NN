use plotters::{
    prelude::{BitMapBackend, CandleStick, IntoDrawingArea},
    style::{Color, BLACK, BLUE, GREEN, RED, WHITE},
};

pub fn costs_candle(
    avg_costs: &[f32],
    max_costs: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("charts/costs.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let y_min = *avg_costs
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let y_max = *avg_costs
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    let mut chart = plotters::chart::ChartBuilder::on(&root)
        .caption("Loss per epoch", ("sans-serif", 20))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..avg_costs.len() as u32, y_min..y_max)?;

    chart.configure_mesh().light_line_style(WHITE).draw()?;

    /* chart
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()?; */

    chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("Cost")
        .draw()?;

    // Avg costs

    chart
        .draw_series(avg_costs.iter().enumerate().map(|(index, x)| {
            CandleStick::new(index as u32, 0., 0., 0., *x, GREEN.filled(), RED, 15)
        }))?
        .label("Average Cost")
        .legend(|(x, y)| {
            plotters::element::Rectangle::new([(x, y), (x + 20, y + 10)], GREEN.filled())
        });

    // Max costs

    chart
        .draw_series(max_costs.iter().enumerate().map(|(index, x)| {
            CandleStick::new(index as u32, 0., 0., 0., *x, BLUE.filled(), RED, 15)
        }))?
        .label("Max Cost")
        .legend(|(x, y)| {
            plotters::element::Rectangle::new([(x, y), (x + 20, y + 10)], BLUE.filled())
        });

    root.present()
        .expect("unable to write chart to file, perhaps there is no directory");

    Ok(())
}
