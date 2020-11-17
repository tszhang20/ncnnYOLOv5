package com.example.ncnnyolov5;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class YOLOv5 {
    // 初始化函数
    public native boolean Init(AssetManager mgr);
    // 目标类
    public class Obj
    {
        public float x;
        public float y;
        public float w;
        public float h;
        public String label;
        public float prob;
    }
    // 检测函数
    public native Obj[] Detect(Bitmap bitmap, boolean use_gpu);

    static {
        System.loadLibrary("ncnnyolov5");
    }
}
