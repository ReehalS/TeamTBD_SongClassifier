# TeamTBD Song Genre Classification

## Project Overview
This project utilizes data from the Million Song Dataset and GTZAN to classify songs into different genres. By working with mel spectrograms and various audio features such as Tempo, ZCR, Rolloff, Mean, Variance, and Chroma STFT, we aimed to build a robust genre classifier.

## Contributors

- [ReehalS](https://github.com/ReehalS)
- [Camila Nino Francia](https://github.com/CCEW)
- [UtkarshNandy](https://github.com/UtkarshNandy)
- [JishyFishy2006](https://github.com/JishyFishy2006)
- [shlokyu16](https://github.com/shlokyu16)

## Data
We combined selected genres from the Million Song Dataset and GTZAN Dataset to ensure a well-rounded and comprehensive dataset. The songs were split into 3-second intervals to increase the dataset size and specificity. Metrics were generated using the Librosa library for audio processing. The data was normalized using min-max scaling to ensure faster convergence and reduce sensitivity to magnitude variations.

## Methodology
We experimented with various models including K-Nearest Neighbors, Neural Networks, Logistic Regression, and Random Forests. Our final model was an Extreme Gradient Booster (XGB), which achieved the highest accuracy. Gradient boosting builds trees sequentially, with each new tree attempting to correct the errors of the previous trees by fitting to the gradient of the loss function.

### Model Details
- **Estimators**: 4000 (number of trees)
- **Learning Rate**: 0.05 (step size towards the minimum of the loss function)
- **Accuracy**: 78% with 4000 estimators and 0.01 learning rate

Despite achieving 82% accuracy with more estimators and a lower learning rate, the larger model size was not feasible for deployment on GitHub and Streamlit. Increasing estimators and lowering the learning rate also increased the risk of overfitting and significantly prolonged training time. Future work includes hyperparameter tuning and optimization through grid search to find the optimal values for estimators and learning rate.

## Outputs
- **Accuracy**: 78% with 4000 estimators and 0.01 learning rate
- **Trade-offs**: Higher estimators and lower learning rate resulted in only a 4% accuracy increase, at the cost of overfitting and longer training times.

## Future Work
Future steps include performing hyperparameter tuning and optimization through grid search to determine the optimal values for estimators and learning rate.

## Demonstration
Here is a short demonstration of our final model [Watch the demonstration](https://drive.google.com/file/d/1AfBmu3aaE7BZtdE4oQyR1ppKb29zYipl/view?usp=sharing). 
You can also test it yourself at [songclassifiers.streamlit.app](https://songclassifiers.streamlit.app)


## License

This project is licensed under the MIT License. See the LICENSE file for details.

