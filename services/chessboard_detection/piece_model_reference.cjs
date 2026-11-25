/** Run the original TensorFlow.js models on RGB cells and save features and probabilities. */
const fs = require("node:fs");
const path = require("node:path");
const tf = require("@tensorflow/tfjs");

/** Read the original topology and weight shards without an HTTP server. */
function modelArtifacts(directory) {
    const model = JSON.parse(fs.readFileSync(path.join(directory, "model.json")));
    const specs = model.weightsManifest.flatMap((group) => group.weights);
    const weights = Buffer.concat(model.weightsManifest.flatMap((group) =>
        group.paths.map((file) => fs.readFileSync(path.join(directory, file)))));
    return tf.io.fromMemory({
        ...model,
        weightSpecs: specs,
        weightData: weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength),
    });
}

/** Write a portable little-endian float32 sequence. */
function writeFloat32(output, values) {
    const bytes = Buffer.alloc(values.length * 4);
    values.forEach((value, index) => bytes.writeFloatLE(value, index * 4));
    fs.writeFileSync(output, bytes);
}

/** Apply the same preprocessing and inference as the web app to each cell. */
async function main(models, samples) {
    await tf.setBackend("cpu");
    const mobileNet = await tf.loadGraphModel(modelArtifacts(path.join(models, "mobilenet")));
    const head = await tf.loadLayersModel(modelArtifacts(path.join(models, "classification_model")));
    for (const cell of JSON.parse(fs.readFileSync(path.join(samples, "cells.json")))) {
        const bytes = fs.readFileSync(path.join(samples, cell.file));
        tf.tidy(() => {
            const pixels = tf.tensor3d(bytes, [cell.height, cell.width, 3], "int32");
            const input = tf.image.resizeBilinear(pixels, [224, 224], true).div(255).expandDims(0);
            const features = mobileNet.predict(input);
            const scores = head.predict(features);
            writeFloat32(path.join(samples, `${cell.name}.tfjs.bin`),
                new Float32Array([...features.dataSync(), ...scores.dataSync()]));
        });
    }
}

main(...process.argv.slice(2)).catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
