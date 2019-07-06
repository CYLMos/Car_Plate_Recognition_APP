package com.cyl.carplaterecognition;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.googlecode.tesseract.android.TessBaseAPI;

import org.opencv.android.OpenCVLoader;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.Date;

import static android.Manifest.permission.CAMERA;
import static android.Manifest.permission.READ_EXTERNAL_STORAGE;
import static android.Manifest.permission.WRITE_EXTERNAL_STORAGE;

public class MainActivity extends AppCompatActivity {

    private Button btOpenCamera;
    private Button btChoseImage;
    private TextView tvMessage;

    public static ImageView ivImageOrigin;  // for origin image
    public static ImageView ivImageCarPlate;  // for car plate
    public static ImageView ivImageTextRegion;  // for text zone

    private Uri imageFile;

    public static final String DATA_PATH = Environment.getExternalStorageDirectory().getAbsolutePath() + File.separator + "Tess";  // image data path
    public static final String TESS_DATA = File.separator + "tessdata";  // tess data path

    public static final int TESS_TWO = 1;
    public static final int KNN = 2;

    private int algoNum;  // the node number, if 1 is tess-two mode, 2 is knn mode

    static{
        if(!OpenCVLoader.initDebug()){
            Log.e("Myapplication", "openCV not loaded");
        }
        else{
            Log.e("Myapplication", "openCV loaded");
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {

        // if use camera
        if (requestCode == 120) {
            if (resultCode == RESULT_OK) {

                //prepareTessData();

                TessBaseAPI tessBaseAPI = new TessBaseAPI();

                // initial tessBaseAPI
                tessBaseAPI.init(DATA_PATH, "eng");

                // set PSM mode
                //tessBaseAPI.setPageSegMode(TessBaseAPI.PageSegMode.PSM_SINGLE_LINE);

                try{
                    BitmapFactory.Options options = new BitmapFactory.Options();
                    options.inSampleSize = 8;
                    //Bitmap bitmap = BitmapFactory.decodeFile(imageFile.getPath(), options);
                    Bitmap bitmap = (Bitmap) data.getExtras().get("data");

                    ImageProcessAsync imageProcessAsync = new ImageProcessAsync(this, this.tvMessage, this.algoNum);
                    imageProcessAsync.execute(bitmap);

                }catch (Exception e){
                    Log.e("Wrong", e.getMessage());
                }

            }
        }

        // if choose the image from user's phone
        else if(requestCode == 110){
            if(resultCode == RESULT_OK){
                this.imageFile = data.getData();
                Bitmap bitmap = null;
                try {
                    final InputStream imageStream = getContentResolver().openInputStream(this.imageFile);
                    BitmapFactory.Options options = new BitmapFactory.Options();
                    options.inSampleSize = 8;
                    bitmap = BitmapFactory.decodeStream(imageStream, null, options);
                }
                catch (Exception e){
                    Log.e("Wrong", e.getMessage());
                }



                ImageProcessAsync imageProcessAsync = new ImageProcessAsync(this, this.tvMessage, this.algoNum);
                imageProcessAsync.execute(bitmap);
            }
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        this.algoNum = TESS_TWO;

        initView();

        if(!hasPermissions()){
            getPermission();
        }

        prepareTessData();
    }

    // Initial views
    private void initView(){
        this.btOpenCamera = (Button)findViewById(R.id.bt_OpenCamera);
        this.btChoseImage = (Button)findViewById(R.id.bt_ChoseImage);
        this.tvMessage = (TextView)findViewById(R.id.tv_Message);
        this.ivImageOrigin = (ImageView)findViewById(R.id.iv_ImageOrigin);
        this.ivImageCarPlate = (ImageView)findViewById(R.id.iv_ImageCarPlate);
        this.ivImageTextRegion = (ImageView)findViewById(R.id.iv_ImageTextRect);

        this.btOpenCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                imageFile = Uri.fromFile(getOutputMediaFile());

                //intent.putExtra(MediaStore.EXTRA_OUTPUT, imageFile);

                startActivityForResult(intent, 120);
            }
        });

        this.btChoseImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent, "Select Picture"), 110);
            }
        });
    }

    // get dangerous permissions
    private void getPermission(){
        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]{WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE, CAMERA}, 100);
        }
    }

    // check permissions
    private boolean hasPermissions(){
        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
            if(checkSelfPermission(WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED &&
                    checkSelfPermission(READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED &&
                    checkSelfPermission(READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED){

                return true;
            }
            else{
                return false;
            }
        }

        return true;
    }

    // get the image that user use camera to take
    private static File getOutputMediaFile(){
        File mediaStorageDir = new File(Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_PICTURES), "TessImage");

        if (!mediaStorageDir.exists()){
            if (!mediaStorageDir.mkdirs()){
                return null;
            }
        }

        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        return new File(mediaStorageDir.getPath() + File.separator +
                "IMG_"+ timeStamp + ".jpg");
    }

    // if no directories and traindata that app needs, create them
    private void prepareTessData(){
        try{
            File dir = new File(DATA_PATH + TESS_DATA);
            if(!dir.exists()){
                dir.mkdirs();
            }

            String fileList[] = getAssets().list("");
            for(String fileName : fileList){
                String pathToDataFile = DATA_PATH + TESS_DATA + "/" + fileName;

                if(!(new File(pathToDataFile).exists())){
                    InputStream ins = getAssets().open(fileName);
                    OutputStream outs = new FileOutputStream(pathToDataFile);

                    byte [] buff = new byte[1024];
                    int len;

                    while((len = ins.read(buff)) > 0){
                        outs.write(buff, 0, len);
                    }

                    ins.close();
                    outs.close();
                }
            }
        }catch (Exception e){
            Log.e("Wrong", e.getMessage());
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults){
        if(requestCode == 200){
            for(int i = 0; i < permissions.length; i++) {
                String resultString = permissions[i] + (grantResults[i] == PackageManager.PERMISSION_GRANTED ? "Pass" : "Denied");
                Toast.makeText(this, resultString, Toast.LENGTH_SHORT).show();
            }
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu){
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item){
        item.setChecked(!item.isChecked());

        if(item.getItemId() == R.id.rb_knn){
            this.algoNum = KNN;
            Toast.makeText(this, "Knn", Toast.LENGTH_SHORT).show();
        }
        else if(item.getItemId() == R.id.rb_tt){
            this.algoNum = TESS_TWO;
            Toast.makeText(this, "Tess-two", Toast.LENGTH_SHORT).show();
        }

        return super.onOptionsItemSelected(item);
    }
}
