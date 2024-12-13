use std::error::Error;
use csv::{ReaderBuilder, WriterBuilder};

#[derive(Debug, serde::Deserialize, serde::Serialize)]
pub struct Transaction {
    #[serde(rename = "step")]
    pub step: u32,
    #[serde(rename = "type")]
    pub ttype: String,
    #[serde(rename = "amount")]
    pub amount: f64,
    #[serde(rename = "nameOrig")]
    pub name_orig: String,
    #[serde(rename = "oldbalanceOrg")]
    pub old_balance_orig: f64,
    #[serde(rename = "newbalanceOrig")]
    pub new_balance_orig: f64,
    #[serde(rename = "nameDest")]
    pub name_dest: String,
    #[serde(rename = "oldbalanceDest")]
    pub old_balance_dest: f64,
    #[serde(rename = "newbalanceDest")]
    pub new_balance_dest: f64,
    #[serde(rename = "isFraud")]
    pub is_fraud: u8,
    #[serde(rename = "isFlaggedFraud")]
    pub is_flagged_fraud: u8,
}

pub fn clean_data(dir: &str, output: &str) -> Result<(), Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(dir)?;

    let mut writer = WriterBuilder::new()
        .has_headers(true)
        .from_path(output)?;

    // remain header for new csv
    writer.write_record(&[
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    ])?;
    //store for second (1) column for future read and update
    let valid_types = vec!["CASH-IN", "CASH-OUT", "DEBIT", "PAYMENT", "TRANSFER"];

    for result in reader.deserialize::<Transaction>() {
        let mut rec = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        // if type is one of the listed legal type
        if !valid_types.contains(&rec.ttype.as_str()) {
            continue; // erase if the row doesn't follow the criteria
        }
        //change to 1 in last column if amount is over 200,000
        if rec.amount > 200_000.0 {
            rec.is_flagged_fraud = 1;
        }
        // names - does it start with M or C??
        if !(rec.name_orig.starts_with('M') || rec.name_orig.starts_with('C')) {
            continue;
        }
        if !(rec.name_dest.starts_with('M') || rec.name_dest.starts_with('C')) {
            continue;
        }
        // final - cleaned data
        writer.serialize(rec)?;
    }

    writer.flush()?;
    Ok(())
}

pub fn count_rows(dir: &str) -> Result<usize, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)//don't count header
        .from_path(dir)?;
    let mut row_count = 0;

    for result in reader.records() {
        if result.is_ok() {
            row_count += 1;
        }
    }

    Ok(row_count)
}

#[test]
fn test_clean_data_runs() {
    use tempfile::NamedTempFile;
    use std::io::Write;

    let mut input_file = NamedTempFile::new().unwrap();
    let output_file = NamedTempFile::new().unwrap();
    
    writeln!(
        input_file,
        "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud\n\
         1,CASH-IN,100.0,M123,0,100,C456,0,100,0,0"
    )
    .unwrap();

//clean data with temp files
    let result = clean_data(
        input_file.path().to_str().unwrap(),
        output_file.path().to_str().unwrap(),
    );

    // Assert the function runs without error
    assert!(result.is_ok());
}
