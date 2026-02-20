# PathBench

PathBench is a benchmark designed to evaluate pathological speech assessment systems.

## PathBench

## Usage guide

There are several use cases for PathBench:

* [I want to evaluate my newly developed predictor](https://www.google.com/search?q=%23i-want-to-evaluate-my-newly-developed-predictor)

* [I want to use the predictors developed by you](https://www.google.com/search?q=%23i-want-to-use-the-predictors-developed-by-you)

* [I want to contribute a new predictor to this repository, how do I do that?](https://www.google.com/search?q=%23i-want-to-contribute-a-new-predictor-to-this-repository-how-do-i-do-that)

* [I want to reproduce your research](https://www.google.com/search?q=%23i-want-to-reproduce-your-research)

### I want to evaluate my newly developed predictor

For this use case no install needed, it is sufficient if you provide an appropriate CSV file with your predicted scores, and run `csv_evaluation`.

### I want to use the predictors developed by you

You have to install the framework but not download the stuff (datasets).

### I want to contribute a new predictor to this repository, how do I do that?

Please first look at the flow-chart below to decide where the model lies in this taxonomy. If you feel that your model doesn't fit this taxonomy, please open an issue, and we can discuss.

### I want to reproduce your research

Install and download everything.

## How do I download the required datasets?

We are not allowed to share these datasets ourselves, however, all of them are relatively easily accesible. Please get your copy.

* [COPAS](https://taalmaterialen.ivdnt.org/download/tstc-corpus-pathologische-en-normale-spraak-copas/)

* [EasyCall](http://neurolab.unife.it/easycallcorpus/)

* [TORGO](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)

* [NeuroVoz](https://zenodo.org/records/10777657)

* [UASpeech](https://speechtechnology.web.illinois.edu/uaspeech/)

* Oral Cancer - YouTube

## Setup

Please route all wav.scp-s to your root.

## Quick start

The following code shows how to evaluate a dataset and get the correlation of each metric with the ground truth scores.

## Installation

To install the package, you can do the following: