package com.cyl.carplaterecognition;

import android.graphics.Bitmap;

import org.opencv.core.Mat;

/**
 * A interface for recognition algorithms
 */

public interface AlgorithmInterface {

    void train();  // train method

    String getResult();  // return string result

    void setImage(Bitmap bitmap);  // set the bitmap image

    void setImage(Mat mat);  // set th3 mat image
}
