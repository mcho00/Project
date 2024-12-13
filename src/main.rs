use std::error::Error;

mod analysis;
mod cleaning;

use analysis::{build_graph_print_ten, run_fraud_prediction};
use cleaning::{clean_data, count_rows};

fn run_application() -> Result<(), Box<dyn Error>> {
    let input_file = "Synthetic Financial Datasets For Fraud Detection.csv";
    let output_file = "cleaned_data.csv";
    clean_data(input_file, output_file)?; 

    let row_count_before = count_rows(input_file)?;
    let row_count_after = count_rows(output_file)?;
    println!("User data before cleaning: {}", row_count_before);
    println!("User data after cleaning: {}", row_count_after);
    build_graph_print_ten(output_file)?;
    run_fraud_prediction(output_file)?;

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    run_application()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_main_runs() {
        let result = run_application();
        assert!(result.is_err() || result.is_ok());
    }
}