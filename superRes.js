
class L1 {

    static className = 'L1';

    constructor(config) {
       return tf.regularizers.l1l2(config)
    }
}
tf.serialization.registerClass(L1);

//tf.ENV.set('WEBGL_PACK', false)
//tf.setBackend('cpu');
//processor.doLoad();

async function dos()
{
  console.log(tf.getBackend())
  let model = await tf.loadLayersModel("./model.json");

  let c1 = document.getElementById("my-canvas");
  let img = document.getElementById("WTFAREYOUDOING");
  
  console.log("Lol")
  //this.video.pause();
  let framey = tf.browser.fromPixels(img, 3);
  framey = framey.mul(1/255)
  let framey2 = framey.expandDims(0);
  console.log(framey2);
  console.log(framey2.dataSync()[0]);
  console.log(framey2.dataSync()[1000]);
  
  let prediction = model.predict(framey2);
  console.log("predicted")
  console.log(prediction)
  console.log(prediction.squeeze(0).dataSync()[0])
  console.log(prediction.squeeze(0).dataSync()[1000])
  await tf.browser.toPixels(prediction.squeeze(0).minimum(1.0), c1);
  console.log(tf.memory())
  
}



dos();
