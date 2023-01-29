const express = require('express'); // import express lib
const path = require('path');

const hostname = '127.0.0.1';
const port = 3000; // set port to connect to server on  

const app = express(); // create the express server

app.use(express.static('dist')); // tell it where to find the public contents

// app.set("views", "./src") // tell it where to find the views

// Serve the model files
app.use('/tfjs_artifacts', express.static(path.join(__dirname, 'web_model')))


app.get('/', function(req, res) {
  res.sendFile(path.join(__dirname + "/" + 'index.html'));
})

app.listen(port, hostname, () => {
  console.log(`Server running at http://${hostname}:${port}/`);
})