package main

import (
	"fmt"
	"log"

	"self-ai/internal/models"
	"self-ai/internal/utils"

	"gorgonia.org/gorgonia"
)

var (
	dataSets = "./internal/utils/datasets/sub0-4/sub-01/ses-mri/func/sub-01_ses-mri_task-facerecognition_run-01_events.tsv"
)

func main() {
	// Load and preprocess EEG data
	data, err := utils.LoadData(dataSets)
	if err != nil {
		log.Fatal(err)
	}

	// Split data into training and testing sets
	trainData, testData, err := utils.SplitData(data, 0.8) // 80% for training, 20% for testing
	if err != nil {
		log.Fatal("SplitData: ", (err))
	}

	// Define your neural network architecture
	model := &models.Model{
		Graph: gorgonia.NewGraph(),
	}
	model = models.DefineNetwork()

	// Train the model
	model.Train(model.Graph, trainData)

	// Evaluate the model on test data
	accuracy := model.Evaluate(testData)
	fmt.Printf("Accuracy: %.2f%%\n", accuracy*100)

	// Make predictions with the trained model
	sample := utils.GetSampleData()
	prediction := model.Predict(sample)
	fmt.Println("Prediction:", prediction)
}
