# MSci_project

# Overview

The Jelfs group has recently developed Web-BO, a Graphical User Interface (GUI) to support Bayesian Optimisation for chemical optimisation tasks. Web-BO is a modular platform that allows users without any coding experience to easily apply BO algorithms to chemical optimisation applications. Presently, Web-BO supports single-objective chemical optimisation tasks – where one parameter is being optimised sequentially.

This project aims to further develop and maintain Web-BO by extending its capability in two key areas: i) enabling multi-objective optimisation tasks, allowing for optimisation over more than one parameter (e.g. maximising yield, whilst limiting the cost of reagents) ii) introducing additional surrogate model options (i.e. incorporating additional ML models within the framework). 

We will demonstrate the new functionality via several materials optimisation case studies, including for property optimisation and materials design.

# Project steps
## Phase 1 - Research of background literature:
   * Bayesian optimisation basics
   * GitHub and code basics

This first phase will primarily involve compiling a literature base to gain a deeper understanding of the  Bayesian Optimisation process, including the underlying maths. This will include researching different surrogate models and acquisition functions. Additionally, I will become familiar with the code which currently underpins Web-BO.

## Phase 2 - Improvement of Web-BO user experience:
   * Familiarisation with Web-BO current functionality
   * Identification of 3 features to improve
   * Implementation of enhanced features
   * Selection of one or more chemistry-based case studies to test functionality

The second phase will be a general exploration of Web-BO in its current form to evaluate its current functionality and identify areas for improvement. This phase will largely focus on enhancing the user experience of Web-BO.

Further, several chemistry and/or materials-based case studies will be selected to allow for subsequent testing of Web-BO's functionality. We will likely leverage the open-source framework Summit,  which provides several ‘virtual-laboratory’ chemical optimisation benchmarks to evaluate machine learning model performance without performing expensive experiments. The first benchmark offers a four-dimensional multi-objective optimisation example for the nucleophilic substitution reaction (SNAr) between difluoronitrobenzene and pyrrolidine. The optimisation targets a maximisation of space time yield while minimising the E-factor (the ratio of product to waste production).

## Phase 3 - Extension of Web-BO functionality:
   * Implement multi-objective optimisation
   * Introduce additional surrogate models

This phase will extend the functionality of Web-BO to incorporate multi-objective optimisation, enabling the optimisation of more than one target simultaneously, such as maximising yield whilst keeping reactant costs low. Currently Web-BO uses gaussian processes as its surrogate model via the bayBE back-end, but different datasets may benefit from alternative surrogate models. To address this, additional surrogate models will be integrated into Web-BO to facilitate more accurate approximations of objective functions in these cases.

## Phase 4 - Testing stage:
   * Finalise chosen case studies
   * Test Web-BO via case study

Using the case studies selected in phase 2, the performance of Web-BO will be evaluated relative to alternative optimisation methods.  

## Gantt Chart

![8BA8AEB7-D014-4B1C-9347-6600C0462E75](https://github.com/user-attachments/assets/9b85a61e-a444-4897-8f0e-ddc64d51118e)


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
