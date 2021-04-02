
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

  vid.width = newsizeinput.value;
  vid.height = newsizeinput.value;
  c1.width = newsizeinput.value;
  c1.height = newsizeinput.value;
}

var processor = {
    timerCallback: function() {
      if (this.video.paused || this.video.ended) {
        return;
      }
      this.computeFrame()
      var self = this;
      setTimeout(function () {
        self.timerCallback();
      }, 32); // roughly 60 frames per second
    },
  
    doLoad: async function() {
      this.video = document.getElementById("my-video");
      this.c1 = document.getElementById("my-canvas");
      this.ctx1 = this.c1.getContext("2d");
      this.model = await tf.loadLayersModel("./model.json")
      var self = this;
  
      this.video.addEventListener("play", function() {
        self.width = self.video.width;
        self.height = self.video.height;
        self.timerCallback();
      }, false);
    },
    computeFrame: function() {
        this.ctx1.drawImage(this.video, 0, 0, this.width, this.height);
        var frame = this.ctx1.getImageData(0, 0, this.width, this.height);

        //console.log(tf.getBackend())
        
        //console.log("Lol")
        //this.video.pause();
        let framey = tf.browser.fromPixels(frame, 3);
        framey = framey.mul(1/255)
        let framey2 = framey.expandDims(0);
        //console.log(framey2);
        //console.log(framey2.dataSync()[0]);
        //console.log(framey2.dataSync()[1000]);
        
        let prediction = this.model.predict(framey2);
        //console.log("predicted")
        //console.log(prediction)
        //console.log(prediction.squeeze(0).dataSync()[0])
        //console.log(prediction.squeeze(0).dataSync()[1000])
        tf.browser.toPixels(prediction.squeeze(0).minimum(1.0), this.c1);
        //console.log(tf.memory())
        
        prediction.dispose()
        framey.dispose()
        framey2.dispose()


    }
};   



