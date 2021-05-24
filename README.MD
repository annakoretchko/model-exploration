

<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />

<p align="center">

  <h3 align="center">Model Exploration</h3>

  <p align="center">
    Test various aspects of model performance on multiple datasets
    <br />
    <a href="https://connect.garmin.com/"><strong>Garmin Connect »</strong></a>
    <br />
    <br />
    <a href="/dist/garmin_analysis-0.1-py3-none-any.whl">Latest Package</a>
    ·
    <a href="https://anna-koretchko.ue.r.appspot.com/garmin">Demo</a>

  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Projects</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
        <li>
      <a href="#about-the-project">The Projects</a>
      <ul>
        <li><a href="#built-with">OOP_forest</a></li>
        <li><a href="#built-with">classifier_diabetes</a></li>
        <li><a href="#built-with">comparing_classifiers</a></li>
        <li><a href="#built-with">randomforest_v_svm_classifier</a></li>
        <li><a href="#built-with">speed_v_accuracy_multiple_models</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>






<!-- ABOUT THE PROJECT -->
## About The Projects

One of the best ways to learn about modeling, is to just mess around with all models. In this repo, I have mutliple datasources and scripts written to evaluate different model performances. 

The goal of this repo is to employ different models, hyper-tuning techniques, and software structure. You can see each project is based off of comapring models, to one extent or another, but it goes another level deeper by having each project strucutred differently (procedural vs object oriented etc) as a way for me to practice and show just how much analytics is an art. 


I have created both juypter notebooks and python scripts of the code. I personally like pure python code, but for tutorials such as this, jupyter notebooks are a great resource for shraing. The HTML and PDF of the completed notebook is also provided for your own edification/if you don't want to run the code yourself. 

All data used is for the projects can be found in the *data* folder. 

**Enjoy!**


### Built With

This section should list any major frameworks that you built your project using. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [Python](https://www.python.org/)
* [jupyter](https://jupyter.org/)




<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

Python is the only requirement for this project (however, enviroment managers such as conda are greatly suggested as well)
* python
  ```sh
  sudo apt-get install python3.6
  ```

### Installation and Example Run

1. Create conda environment for garmin_analyis project ([install conda Here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html))

2. Downdload lastest wheel (whl) from from the dist folder [here] (/dist/garmin_analysis-0.1-py3-none-any.whl)

3. cd to the downloaded wheel file
   ```sh
   (garmin_env) cd Downloads/garmin_analysis-0.X-py3-none-any.whl
   ```
4. pip install `garmin_analysis-0.X-py3-none-any.whl` to your newly created conda environment
   ```sh
   (garmin_env) pip install garmin_analysis-0.X-py3-none-any.whl
   ```
5. Now run the package which requires two inputs: -data and -output_path 
    ```sh
    (garmin_env) python -m garmin_analysis -data "<path to data file from Garmin Connect" -output_path "<path to where you want the output files to be saved"
    ```

<!-- USAGE EXAMPLES -->
## Usage

This packages allows Garmin Connect users to analyize their data in a pythonic manner, which can be extrapolated further into a myriad of projects! 
_To see a live demo of the graphs/outputs, please refer to the [Demo](https://anna-koretchko.ue.r.appspot.com/garmin)_



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

* Anna Koretchko - [Personal Website](https://anna-koretchko.ue.r.appspot.com/index)
* Email - annakoretchko@gmail.com







<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/annakoretchko/garmin_analysis/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/anna-koretchko-1b5b0211a/
[product-screenshot]: images/screenshot.png

