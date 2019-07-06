package com.cyl.carplaterecognition;

import android.graphics.Bitmap;

import com.googlecode.tesseract.android.TessBaseAPI;

import org.opencv.core.Mat;

/**
 * the class of the Tess-two
 */

public class Tesstow implements AlgorithmInterface {

    private TessBaseAPI tessBaseAPI;

    public Tesstow(){
        this.tessBaseAPI = new TessBaseAPI();
    }

    @Override
    public void train() {
        this.tessBaseAPI.init(MainActivity.DATA_PATH, "eng");
        this.tessBaseAPI.setVariable(TessBaseAPI.VAR_CHAR_WHITELIST, "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-");  // white list
        this.tessBaseAPI.setPageSegMode(TessBaseAPI.PageSegMode.PSM_SINGLE_LINE);  // use single line mode
    }

    @Override
    public String getResult() {
        String result = this.tessBaseAPI.getUTF8Text();
        /*String text = this.tessBaseAPI.getUTF8Text();
        String[] sentences = text.split("\n");

        for(int i = 0; i < sentences.length; i++){
            if(sentences[i].trim().length() > result.trim().length()){
                result = sentences[i];
            }
        }*/

        return result;
    }

    @Override
    public void setImage(Bitmap bitmap) {
        this.tessBaseAPI.setImage(bitmap);
    }

    @Override
    public void setImage(Mat mat) {

    }
}
