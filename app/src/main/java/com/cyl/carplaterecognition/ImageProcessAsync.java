package com.cyl.carplaterecognition;

import android.app.AlertDialog;
import android.content.Context;
import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.TextView;

import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.imgproc.*;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * the class of image process
 */

public class ImageProcessAsync extends AsyncTask<Bitmap, Integer , String> {

    //private ProgressBar progressBar;
    private Context context;

    private TextView tvMessage;
    private Bitmap bmOrigin;
    private Bitmap bmCarPlate;
    private Bitmap bmTextRegion;

    private AlertDialog alertDialog;
    private Imgproc imgproc; //new Imgproc();

    private int algoNum = -1;  // knn is 2, tess-two is 1

    public ImageProcessAsync(Context context, TextView tvMessage, int num){
        this.context = context;
        this.tvMessage = tvMessage;
        this.algoNum = num;
        this.imgproc = new Imgproc();

    }

    @Override
    protected String doInBackground(Bitmap... params) {
        this.bmOrigin = params[0];

        Mat mat = new Mat();
        Utils.bitmapToMat(this.bmOrigin, mat);

        this.imgproc.resize(mat, mat, new Size(400, 300));
        this.imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2RGB);



        Mat mat2 = new Mat();
        this.imgproc.bilateralFilter(mat, mat2,60, 120, 10);  // fuzzy the image

        Mat temp = findCarPlate(mat2, 34, 10);  // find the car plate
        Mat temp2 = findTextRect(temp);  // find the word zone from the car plate
        List<Mat> dstMatList = wordFilter(temp2);  // split

        Mat dstMat = new Mat();

        this.imgproc.cvtColor(temp2, dstMat, Imgproc.COLOR_RGBA2GRAY);

        this.imgproc.threshold(dstMat, dstMat, 100, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

        Bitmap testBM = Bitmap.createBitmap(dstMat.cols(), dstMat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(dstMat, testBM);

        this.bmCarPlate = Bitmap.createBitmap(temp.cols(), temp.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(temp, this.bmCarPlate);

        this.bmTextRegion = Bitmap.createBitmap(temp2.cols(), temp2.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(temp2, this.bmTextRegion);


        String message = "";

        AlgorithmInterface algorithmInterface = null;

        if(this.algoNum == MainActivity.TESS_TWO){
            algorithmInterface = new Tesstow();
        }
        else if(this.algoNum == MainActivity.KNN){
            algorithmInterface = new Knn(this.context);
        }

        if(algorithmInterface != null) {
            algorithmInterface.train();

            if(this.algoNum == MainActivity.TESS_TWO){
                algorithmInterface.setImage(testBM);
            }
            else if(this.algoNum == MainActivity.KNN){
                for (int i = 0; i < dstMatList.size(); i++) {
                    algorithmInterface.setImage(dstMatList.get(i));
                }
            }

            message = algorithmInterface.getResult();
        }

        temp.release();
        mat.release();
        temp2.release();


        return message;
    }

    @Override
    protected void onPreExecute(){
        AlertDialog.Builder builder = new AlertDialog.Builder(this.context);
        View view = LayoutInflater.from(this.context).inflate(R.layout.async_dialog, null);
        builder.setView(view);

        this.alertDialog = builder.create();

        this.alertDialog.show();
    }

    @Override
    protected void onPostExecute(String result){
        this.alertDialog.dismiss();
        this.alertDialog.cancel();

        this.tvMessage.setText(result);
        MainActivity.ivImageOrigin.setImageBitmap(this.bmOrigin);
        MainActivity.ivImageCarPlate.setImageBitmap(this.bmCarPlate);
        MainActivity.ivImageTextRegion.setImageBitmap(this.bmTextRegion);
    }

    private void setAlgoNum(int num){
        this.algoNum = num;
    }

    // split every char
    private List<Mat> wordFilter(Mat mat){
        List<Mat> matList = new ArrayList<>();
        List<Rect> rectList = new ArrayList<>();
        Mat dstMat = new Mat();
        Mat dstTemp = new Mat();

        this.imgproc.cvtColor(mat, dstMat, Imgproc.COLOR_RGBA2GRAY);  // become the gray image

        this.imgproc.threshold(dstMat, dstTemp, 100, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);  // become the binary image

        List<MatOfPoint> contours = new ArrayList<>();

        this.imgproc.findContours(dstTemp, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        //Scalar color = new Scalar(255, 0, 255);

        for(int i = 0; i < contours.size(); i++){
            Rect rect = Imgproc.boundingRect(contours.get(i));
            boolean sizeFlag = rect.height > (mat.height()/3) && rect.width < (mat.width()/4) && rect.width > 0 ? true : false;  // if the size is a char
            if(sizeFlag){
                rectList.add(rect);
            }
            //this.imgproc.rectangle(dstMat, new Point(rect.x, rect.y), new Point(rect.x+rect.width, rect.y+rect.height), color, 4);
        }

        // sort them
        Collections.sort(rectList, new Comparator<Rect>() {
            @Override
            public int compare(Rect lrect, Rect rrect) {
                return lrect.x < rrect.x ? -1 : 1 ;
            }
        });

        for(int i = 0; i < rectList.size(); i++){
            Mat chMat = dstTemp.submat(rectList.get(i));
            /*this.imgproc.rectangle(mat, new Point(rectList.get(i).x, rectList.get(i).y),
                    new Point(rectList.get(i).x+rectList.get(i).width, rectList.get(i).y+rectList.get(i).height),
                    color, 4);*/

            matList.add(chMat);
        }
        rectList.clear();

        /*this.bitmap.recycle();
        this.bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat, this.bitmap);*/

        return matList;
    }

    // find the text zone
    private Mat findTextRect(Mat mat){
        List<Rect> rectList = new ArrayList<>();
        Mat dstMat = new Mat();

        this.imgproc.cvtColor(mat, dstMat, Imgproc.COLOR_RGBA2GRAY);  // become the gray image

        this.imgproc.threshold(dstMat, dstMat, 100, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);  // become the binary image

        List<MatOfPoint> contours = new ArrayList<>();

        this.imgproc.findContours(dstMat, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        //Scalar color = new Scalar(255, 0, 255);

        Rect yhRect = null;  // store the char image that y is smallest
        Rect ylRect = null;  // store the char image that y is largest

        for(int i = 0; i < contours.size(); i++){
            Rect rect = Imgproc.boundingRect(contours.get(i));
            boolean sizeFlag = rect.height > (mat.height()/3) && rect.width < (mat.width()/4) && rect.width > 0 ? true : false;  // if the size is a char
            if(sizeFlag){
                rectList.add(rect);
                if(rectList.size() != 1){
                    if(rect.y + rect.height < yhRect.y){
                        yhRect = rect;
                    }
                    if(rect.y > ylRect.y){
                        ylRect = rect;
                    }
                }
                else{
                    yhRect = ylRect = rect;
                }
            }
            //this.imgproc.rectangle(dstMat, new Point(rect.x, rect.y), new Point(rect.x+rect.width, rect.y+rect.height), color, 4);
        }


        // sort them
        Collections.sort(rectList, new Comparator<Rect>() {
            @Override
            public int compare(Rect lrect, Rect rrect) {
                return lrect.x < rrect.x ? -1 : 1 ;
            }
        });

        Rect textRect = null;
        Mat output = null;

        if(rectList.size() >= 6) {
            textRect = new Rect(new Point(rectList.get(0).x, yhRect.y),
                    new Point(rectList.get(rectList.size() - 1).x + rectList.get(rectList.size() - 1).width, ylRect.y + ylRect.height));
            output = mat.submat(textRect);
        }
        else{
            output = mat;
        }

        rectList.clear();

        return output;
    }

    // find car plate function
    private Mat findCarPlate(Mat mat, int closeSizeW, int closeSizeH){
        Mat dstMat = mat.clone();//new Mat();

        this.imgproc.cvtColor(dstMat, dstMat, Imgproc.COLOR_RGBA2GRAY);

        dstMat = sobel(dstMat);

        dstMat = binaryImg(dstMat);

        dstMat = close(dstMat, closeSizeW, closeSizeH);

        dstMat = findPlate(mat, dstMat);

        return dstMat;
    }

    private Mat findPlate(Mat mat, Mat dst){
        List<MatOfPoint> contours = new ArrayList<>();

        this.imgproc.findContours(dst, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        Rect carPlateRect = null;
        //Scalar color = new Scalar(255, 0, 255);
        for(int i = 0; i < contours.size(); i++) {
            Rect rect = Imgproc.boundingRect(contours.get(i));

            boolean shapeFlag = rect.width > rect.height ? true : false;  // if the rectangle width is bigger than height

            if (i != 0 && shapeFlag) {
                boolean sizeFlag = (double) (rect.width * rect.height) > (double) (carPlateRect.width * carPlateRect.height) ? true : false;  // if size matches
                if (sizeFlag) {
                    carPlateRect = rect;
                }
            }
            else if (i == 0) {
                carPlateRect = rect;
            }
        }

        Mat output = mat.submat(carPlateRect);
        this.imgproc.resize(output, output, new Size(600, 300));  // resize the image

        return output;
    }

    // dilate and erode
    private Mat close(Mat mat, int closeSizeW, int closeSizeH){
        Mat out = new Mat();

        Mat element = this.imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(closeSizeW, closeSizeH));

        this.imgproc.morphologyEx(mat, out, Imgproc.MORPH_CLOSE, element);
        //this.imgproc.dilate(mat, out, element/*new Mat(3, 3, CvType.CV_8U)*/);
        //this.imgproc.erode(mat, out, element);

        return out;
    }

    // binary function
    private Mat binaryImg(Mat mat){
        Mat out = new Mat();
        //this.imgproc.adaptiveThreshold(mat, out, 255, imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 15, 0);
        this.imgproc.threshold(mat, out, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);

        return out;
    }

    // sobel function
    private Mat sobel(Mat mat){
        this.imgproc.Sobel(mat, mat, CvType.CV_8UC1, 1, 0, 3, 1, 0, Core.BORDER_DEFAULT );

        Mat out = new Mat();
        Core.convertScaleAbs(mat, out);

        return out;
    }


    // not use
    private char charCheck(int index, float label){
        char charLabel = (char)( (int)label );
        if(index < 3){
            if(charLabel == '8'){
                charLabel = 'B';
            }
            else if(charLabel == '0'){
                charLabel = 'G';
            }
        }

        else{
            if(charLabel == 'H' || charLabel == 'B'){
                charLabel = '8';
            }
            else if(charLabel == 'G'){
                charLabel = '0';
            }
        }

        return charLabel;
    }
}
