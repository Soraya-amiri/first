# first

A small starter repository named "first" that demonstrates a basic project structure.

## Description

This repository now contains a minimal example of forecasting daily prices with
Python.  A synthetic dataset of historical prices is provided along with a
script that trains a polynomial regression model and produces a one-week
forecast.

## Usage

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the example forecaster:

   ```bash
   python -m src.price_forecasting
   ```

   The script will print the hold-out mean absolute error and a table of the
   next seven days of predicted prices based on the sample dataset in
   `data/price_history.csv`.

## Contributing

Contributions are welcome. Open issues or pull requests to propose changes.

## License

Specify a license for your project (for example, MIT). If you don't know which
one to use, see https://choosealicense.com/.

## Author

Soraya-amiri
