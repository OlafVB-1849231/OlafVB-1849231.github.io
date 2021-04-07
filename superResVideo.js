var colorconverter = colorconv

standardwidth = 128;
standardheight = 72;

class L1 {

    static className = 'L1';

    constructor(config) {
       return tf.regularizers.l1l2(config)
    }
}




const SCALE = 2;

const isTensorArray = (inputs) => {
  return Array.isArray(inputs);
};

const getInput = (inputs) => {
  if (isTensorArray(inputs)) {
    return inputs[0];
  }
  return inputs;
};


class PixelShuffle extends tf.layers.Layer {
  scale;

  constructor() {
    super({});
    this.scale = SCALE;
  }

  computeOutputShape(inputShape) {
    return [inputShape[0], inputShape[1], inputShape[2], 3];
  }

  call(inputs) {
    return tf.depthToSpace(getInput(inputs), this.scale, 'NHWC');
  }

  static className = 'PixelShuffle';
}

tf.serialization.registerClass(L1);
tf.serialization.registerClass(PixelShuffle)

function changeSize()
{
  let newsizeinput = document.getElementById("sizeinput");
  let vid = document.getElementById("my-video");
  let c1 = document.getElementById("my-canvas");

  vid.width = standardwidth * newsizeinput.value;
  vid.height = standardheight * newsizeinput.value;
  c1.width = standardwidth * newsizeinput.value;
  c1.height = standardheight * newsizeinput.value;
  width = standardwidth * newsizeinput.value;
  height = standardheight *newsizeinput.value;
  c2.width = standardwidth * newsizeinput.value * 2;
  c2.height = standardheight * newsizeinput.value * 2;
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
  let convert = framey.arraySync();

  var i;
  for (i = 0; i < convert.length; i++) {
    var j;
    for (j=0; j < convert[i].length; j++)
    {
      convert[i][j] = colorconverter.RGB2YUV(convert[i][j]);
    }
  } 

  framey = tf.tensor3d(convert, framey.shape)
  let [y, u, v] = tf.split(framey, 3, 2);

  y = y.mul(1/255)
  y = y.expandDims(0);
  
  let prediction = model.predict(y);
  prediction = prediction.squeeze(0);
  framey = tf.image.resizeBilinear(convert, [prediction.shape[0], prediction.shape[1]]);
  framey = framey.slice([0,0,1]);
  prediction = prediction.mul(255);
  framey = prediction.concat(framey, 2);
  convert = framey.arraySync();
  for (i = 0; i < convert.length; i++) {
    var j;
    for (j=0; j < convert[i].length; j++)
    {
      convert[i][j] = colorconverter.YUV2RGB(convert[i][j]);
    }
  } 

  framey = tf.tensor3d(convert, framey.shape)
  framey = tf.cast(framey, "int32")
  framey = tf.clipByValue(framey, 0, 255)
  tf.browser.toPixels(framey, c2);
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