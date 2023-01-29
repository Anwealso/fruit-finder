/**
 * index.js
 * 
 * Alex Nicholson
 * 14/01/2023
 */

/* -------------------------------------------------------------------------- */
/*                                   IMPORTS                                  */
/* -------------------------------------------------------------------------- */
import * as tf from '/node_modules/@tensorflow/tfjs';
import * as tfd from '/node_modules/@tensorflow/tfjs-data';
import * as ui from './ui.js';


/* -------------------------------------------------------------------------- */
/*                                 GLOBAL VARS                                */
/* -------------------------------------------------------------------------- */
let webcam; // A webcam iterator that generates Tensors from the images from the webcam.
let model; // The imported trained model
let isPredicting = false;


/* -------------------------------------------------------------------------- */
/*                               EVENT LISTENERS                              */
/* -------------------------------------------------------------------------- */
document.getElementById('train').addEventListener('click', async () => {
  isPredicting = false;
});

document.getElementById('predict').addEventListener('click', () => {
  isPredicting = true;
  predict();
});


/* -------------------------------------------------------------------------- */
/*                                  FUNCTIONS                                 */
/* -------------------------------------------------------------------------- */
async function runInference() {
  /**
   * Gets images from the webcam and runs the model on them to get the model's 
   * predictions
   */

  ui.isPredicting();
  while (isPredicting) {
    // Capture the frame from the webcam.
    const img = await getImage();
    console.log(img);

    // Make a prediction through mobilenet
    // const predictions = model.executeAsync(img);
    let predictions = await model.executeAsync(
      { 'input_tensor' : img },
      [ 'detection_boxes','detection_scores','detection_classes','num_detections']);
    console.log(predictions);

    // Returns the index with the maximum probability. This number corresponds
    // to the class the model thinks is the most probable given the input.
    // const predictedClass = predictions.as1D().argMax();
    // const classId = (await predictedClass.data())[0];
    
    // // console.log(predictedClass);
    // console.log(classId);

    img.dispose();

    // ui.predictClass(classId);
    await tf.nextFrame();
  }
  ui.donePredicting();
}

async function getImage() {
  /**
   * Captures a frame from the webcam and normalizes it between -1 and 1.
   * Returns a batched image (1-element batch) of shape [1, w, h, c].
   */
  
  const img = await webcam.capture();
  const processedImg =
      tf.tidy(() => img.expandDims(0));
  img.dispose();
  return processedImg;
}


/* -------------------------------------------------------------------------- */
/*                                MAIN FUNCTION                               */
/* -------------------------------------------------------------------------- */
async function init() {
  webcamConfig = {};
  webcamConfig.facingMode = 'environment'; // show the front facing camera on mobile devices

  try {
    webcam = await tfd.webcam(document.getElementById('webcam'), webcamConfig);
  } catch (e) {
    console.log(e);
    document.getElementById('no-webcam').style.display = 'block';
  }
  model = await tf.loadGraphModel('http://127.0.0.1:3000/tfjs_artifacts/model.json');

  ui.init();

  // Warm up the model. This uploads weights to the GPU and compiles the WebGL
  // programs so the first time we collect data from the webcam it will be
  // quick.
  const screenShot = await webcam.capture();
  
  // Run the model immediately
  isPredicting = true;
  runInference();
}

// Initialize the application.
init();
