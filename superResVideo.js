
standardwidth = 128;
standardheight = 72;

class L1 {

    static className = 'L1';

    constructor(config) {
       return tf.regularizers.l1l2(config)
    }
}
tf.serialization.registerClass(L1);

function changeSize()
{
  let newsizeinput = document.getElementById("sizeinput");
  let vid = document.getElementById("my-video");
  let c1 = document.getElementById("my-canvas");

  vid.width = standardwidth * newsizeinput.value / 2 ;
  vid.height = standardheight * newsizeinput.value / 2;
  c1.width = standardwidth * newsizeinput.value;
  c1.height = standardheight * newsizeinput.value;
  width = standardwidth * newsizeinput.value;
  height = standardheight *newsizeinput.value;
  c2.width = standardwidth * newsizeinput.value;
  c2.height = standardheight * newsizeinput.value;
}

let video = document.getElementById("my-video");
let c1 = document.getElementById("my-canvas");
let c2 = document.getElementById("my-canvas2");
let ctx1 = c1.getContext("2d");
let model = null;
tf.loadLayersModel("./model.json").then((value) => {model = value; console.log(model);});
let width = c1.width;
let height = c1.height;
let date = new Date();
let lastFrame = date.getTime();

function computeFrame()
{
  ctx1.drawImage(video, 0, 0, width, height);
  var frame = ctx1.getImageData(0, 0, width, height);

  //console.log(tf.getBackend())
  
  //console.log("Lol")
  //this.video.pause();
  let framey = tf.browser.fromPixels(frame, 3);
  framey = framey.mul(1/255)
  let framey2 = framey.expandDims(0);
  //console.log(framey2);
  //console.log(framey2.dataSync()[0]);
  //console.log(framey2.dataSync()[1000]);
  
  let prediction = model.predict(framey2);
  //console.log("predicted")
  //console.log(prediction)
  //console.log(prediction.squeeze(0).dataSync()[0])
  //console.log(prediction.squeeze(0).dataSync()[1000])
  tf.browser.toPixels(prediction.squeeze(0).minimum(1.0), c2);
  console.log(tf.memory())
  //video.pause();
}

function timerCallback() {
  if (video.paused || video.ended) {
    return;
  }
  tf.tidy(computeFrame);
  let currentdata = new Date();
  let currenttime = currentdata.getTime();
  //console.log(currenttime)
  console.log(1 / ((currenttime - lastFrame) / 1000));
  lastFrame = currenttime

  // setTimeout(function () {
  //   self.timerCallback();
  // }, 32); // roughly 60 frames per second
  window.requestAnimationFrame(timerCallback);
}

video.addEventListener("play", function() {
  timerCallback();
}, false);

changeSize();