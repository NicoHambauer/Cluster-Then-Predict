# cluster-then-predict
The repository contains the code to a benchmark studying different cluster-then-predict (CTP) models. CTP models are a type of model that work in a two staged way. They first cluster samples into smaller subsets and then predict a target variable.


## Environment Setup

We provide a simple script to set up the Conda environment tailored to your operating system. To get started, ensure that Conda is installed on your system and that you have cloned this repository to your local machine.

To set up the Conda environment, execute the following command in the root directory of this project:

```shell
./setup_environment.sh
```

This command will detect your operating system and create a Conda environment with the necessary dependencies for your platform. For macOS users, this will set up an environment that is compatible with Apple Silicon. For Unix/Windows users, the script will include support for cudatoolkit if applicable.

Please make sure to activate the new environment with:

```shell
conda activate CTP
```

Should you face any issues during the setup, please ensure that you have the necessary permissions to execute the script. If needed, you can make the script executable by running:

```shell
chmod +x setup_environment.sh
```

## Datasets

Dataset names are aliased in the code as follows.
All public datasets are available for download.
We did not provide a re-published version of the dataset in this repository directly, but refer the user to these links prio to using these benchmarking datasets.

### Classification
| Dataset name | Alias  | Repository Link |
|:-------------|:-------|-----------------|
| college      | college| https://www.kaggle.com/datasets/saddamazyazy/go-to-college-dataset |
| water        | water  | https://kaggle.com/adityakadiwal/water-potability |
| stroke       | stroke | https://kaggle.com/fedesoriano/stroke-prediction-dataset |
| churn        | telco  | https://kaggle.com/blastchar/telco-customer-churn |
| recidivism   | compas | https://www.kaggle.com/datasets/danofer/compass |
| credit       | fico   | https://community.fico.com/s/explainable-machine-learning-challenge |
| income       | adult  | https://archive.ics.uci.edu/ml/datasets/adult |
| bank         | bank   | https://archive.ics.uci.edu/ml/datasets/Bank+Marketing |
| airline      | airline| https://kaggle.com/teejmahal20/airline-passenger-satisfaction |
| weather      | weather| https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package |

### Regression
| Dataset name | Alias  | Repository Link |
|--------------|--------|-----------------|
| car          | car    | https://archive.ics.uci.edu/ml/datasets/automobile |
| student      | student| https://archive.ics.uci.edu/ml/datasets/Student+Performance |
| productivity | productivity| https://archive.ics.uci.edu/ml/datasets/Productivity+Prediction+of+Garment+Employees |
| insurance    | medical| https://www.kaggle.com/datasets/mirichoi0218/insurance |
| crimes       | crimes | https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime |
| farming      | crab   | https://www.kaggle.com/datasets/sidhus/crab-age-prediction |
| wine         | wine   | https://archive.ics.uci.edu/ml/datasets/wine+quality |
| bike         | bike   | https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset |
| house        | housing| https://www.kaggle.com/datasets/camnugent/california-housing-prices |
| diamond      | diamond| https://www.kaggle.com/datasets/nancyalaswad90/diamonds-prices |
