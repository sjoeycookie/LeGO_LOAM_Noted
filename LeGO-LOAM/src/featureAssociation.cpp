// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following papers:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.
//   T. Shan and B. Englot. LeGO-LOAM: Lightweight and Ground-Optimized Lidar Odometry and Mapping on Variable Terrain
//      IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). October 2018.

#include  "utility.h"


class FeatureAssociation{

private:

	ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;
    ros::Subscriber subLaserCloudInfo;
    ros::Subscriber subOutlierCloud;
    ros::Subscriber subImu;

    ros::Publisher pubCornerPointsSharp;
    ros::Publisher pubCornerPointsLessSharp;
    ros::Publisher pubSurfPointsFlat;
    ros::Publisher pubSurfPointsLessFlat;

    pcl::PointCloud<PointType>::Ptr segmentedCloud;
    pcl::PointCloud<PointType>::Ptr outlierCloud;

    pcl::PointCloud<PointType>::Ptr cornerPointsSharp;
    pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp;
    pcl::PointCloud<PointType>::Ptr surfPointsFlat;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlat;

    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScanDS;

    pcl::VoxelGrid<PointType> downSizeFilter;

    double timeScanCur;
    double timeNewSegmentedCloud;
    double timeNewSegmentedCloudInfo;
    double timeNewOutlierCloud;

    bool newSegmentedCloud;
    bool newSegmentedCloudInfo;
    bool newOutlierCloud;

    cloud_msgs::cloud_info segInfo;
    std_msgs::Header cloudHeader;

    int systemInitCount;
    bool systemInited;

    std::vector<smoothness_t> cloudSmoothness;
    float *cloudCurvature;
    int *cloudNeighborPicked;
    int *cloudLabel;

    int imuPointerFront;
    int imuPointerLast;
    int imuPointerLastIteration;

    float imuRollStart, imuPitchStart, imuYawStart;
    float cosImuRollStart, cosImuPitchStart, cosImuYawStart, sinImuRollStart, sinImuPitchStart, sinImuYawStart;
    float imuRollCur, imuPitchCur, imuYawCur;

    float imuVeloXStart, imuVeloYStart, imuVeloZStart;
    float imuShiftXStart, imuShiftYStart, imuShiftZStart;

    float imuVeloXCur, imuVeloYCur, imuVeloZCur;
    float imuShiftXCur, imuShiftYCur, imuShiftZCur;

    float imuShiftFromStartXCur, imuShiftFromStartYCur, imuShiftFromStartZCur;
    float imuVeloFromStartXCur, imuVeloFromStartYCur, imuVeloFromStartZCur;

    float imuAngularRotationXCur, imuAngularRotationYCur, imuAngularRotationZCur;
    float imuAngularRotationXLast, imuAngularRotationYLast, imuAngularRotationZLast;
    float imuAngularFromStartX, imuAngularFromStartY, imuAngularFromStartZ;

    double imuTime[imuQueLength];
    float imuRoll[imuQueLength];
    float imuPitch[imuQueLength];
    float imuYaw[imuQueLength];

    float imuAccX[imuQueLength];
    float imuAccY[imuQueLength];
    float imuAccZ[imuQueLength];

    float imuVeloX[imuQueLength];
    float imuVeloY[imuQueLength];
    float imuVeloZ[imuQueLength];

    float imuShiftX[imuQueLength];
    float imuShiftY[imuQueLength];
    float imuShiftZ[imuQueLength];

    float imuAngularVeloX[imuQueLength];
    float imuAngularVeloY[imuQueLength];
    float imuAngularVeloZ[imuQueLength];

    float imuAngularRotationX[imuQueLength];
    float imuAngularRotationY[imuQueLength];
    float imuAngularRotationZ[imuQueLength];



    ros::Publisher pubLaserCloudCornerLast;
    ros::Publisher pubLaserCloudSurfLast;
    ros::Publisher pubLaserOdometry;
    ros::Publisher pubOutlierCloudLast;

    int skipFrameNum;
    bool systemInitedLM;

    int laserCloudCornerLastNum;
    int laserCloudSurfLastNum;

    int *pointSelCornerInd;
    float *pointSearchCornerInd1;
    float *pointSearchCornerInd2;

    int *pointSelSurfInd;
    float *pointSearchSurfInd1;
    float *pointSearchSurfInd2;
    float *pointSearchSurfInd3;

    float transformCur[6];
    float transformSum[6];

    float imuRollLast, imuPitchLast, imuYawLast;
    float imuShiftFromStartX, imuShiftFromStartY, imuShiftFromStartZ;
    float imuVeloFromStartX, imuVeloFromStartY, imuVeloFromStartZ;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerLast;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfLast;

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    PointType pointOri, pointSel, tripod1, tripod2, tripod3, pointProj, coeff;

    nav_msgs::Odometry laserOdometry;

    tf::TransformBroadcaster tfBroadcaster;
    tf::StampedTransform laserOdometryTrans;

    bool isDegenerate;
    cv::Mat matP;

    int frameCount;

public:

    FeatureAssociation():
        nh("~")
        {
        // 订阅和发布各类话题
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/segmented_cloud", 1, &FeatureAssociation::laserCloudHandler, this);  //订阅上一步分割的点云（含有地面)
        subLaserCloudInfo = nh.subscribe<cloud_msgs::cloud_info>("/segmented_cloud_info", 1, &FeatureAssociation::laserCloudInfoHandler, this);   //订阅上一步分割信息的点云
        subOutlierCloud = nh.subscribe<sensor_msgs::PointCloud2>("/outlier_cloud", 1, &FeatureAssociation::outlierCloudHandler, this);  //订阅上一步异常的点云
        subImu = nh.subscribe<sensor_msgs::Imu>(imuTopic, 50, &FeatureAssociation::imuHandler, this);  //订阅IMU数据

        pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 1);
        pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 1);
        pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 1);
        pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 1);

        pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2);
        pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2);
        pubOutlierCloudLast = nh.advertise<sensor_msgs::PointCloud2>("/outlier_cloud_last", 2);
        pubLaserOdometry = nh.advertise<nav_msgs::Odometry> ("/laser_odom_to_init", 5);
        
        initializationValue();
    }

    // 各类参数的初始化
    void initializationValue()
    {
        cloudCurvature = new float[N_SCAN*Horizon_SCAN];         //指针开辟对应的堆区
        cloudNeighborPicked = new int[N_SCAN*Horizon_SCAN];
        cloudLabel = new int[N_SCAN*Horizon_SCAN];

        pointSelCornerInd = new int[N_SCAN*Horizon_SCAN];
        pointSearchCornerInd1 = new float[N_SCAN*Horizon_SCAN];
        pointSearchCornerInd2 = new float[N_SCAN*Horizon_SCAN];

        pointSelSurfInd = new int[N_SCAN*Horizon_SCAN];
        pointSearchSurfInd1 = new float[N_SCAN*Horizon_SCAN];
        pointSearchSurfInd2 = new float[N_SCAN*Horizon_SCAN];
        pointSearchSurfInd3 = new float[N_SCAN*Horizon_SCAN];

        cloudSmoothness.resize(N_SCAN*Horizon_SCAN);     //容器开辟空间大小

        downSizeFilter.setLeafSize(0.2, 0.2, 0.2); // 下采样滤波器设置叶子大小 0.2*0.2*0.2

        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        cornerPointsSharp.reset(new pcl::PointCloud<PointType>());
        cornerPointsLessSharp.reset(new pcl::PointCloud<PointType>());
        surfPointsFlat.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlat.reset(new pcl::PointCloud<PointType>());

        surfPointsLessFlatScan.reset(new pcl::PointCloud<PointType>());
        surfPointsLessFlatScanDS.reset(new pcl::PointCloud<PointType>());

        timeScanCur = 0;
        timeNewSegmentedCloud = 0;
        timeNewSegmentedCloudInfo = 0;
        timeNewOutlierCloud = 0;

        newSegmentedCloud = false;
        newSegmentedCloudInfo = false;
        newOutlierCloud = false;

        systemInitCount = 0;
        systemInited = false;

        imuPointerFront = 0;
        imuPointerLast = -1;
        imuPointerLastIteration = 0;

        imuRollStart = 0; imuPitchStart = 0; imuYawStart = 0;
        cosImuRollStart = 0; cosImuPitchStart = 0; cosImuYawStart = 0;
        sinImuRollStart = 0; sinImuPitchStart = 0; sinImuYawStart = 0;
        imuRollCur = 0; imuPitchCur = 0; imuYawCur = 0;

        imuVeloXStart = 0; imuVeloYStart = 0; imuVeloZStart = 0;
        imuShiftXStart = 0; imuShiftYStart = 0; imuShiftZStart = 0;

        imuVeloXCur = 0; imuVeloYCur = 0; imuVeloZCur = 0;
        imuShiftXCur = 0; imuShiftYCur = 0; imuShiftZCur = 0;

        imuShiftFromStartXCur = 0; imuShiftFromStartYCur = 0; imuShiftFromStartZCur = 0;
        imuVeloFromStartXCur = 0; imuVeloFromStartYCur = 0; imuVeloFromStartZCur = 0;

        imuAngularRotationXCur = 0; imuAngularRotationYCur = 0; imuAngularRotationZCur = 0;
        imuAngularRotationXLast = 0; imuAngularRotationYLast = 0; imuAngularRotationZLast = 0;
        imuAngularFromStartX = 0; imuAngularFromStartY = 0; imuAngularFromStartZ = 0;

        for (int i = 0; i < imuQueLength; ++i)     //extern const int imuQueLength = 200，imu队列长度为200
        {
            //imu相关数组初始化全部为0
            imuTime[i] = 0;
            imuRoll[i] = 0; imuPitch[i] = 0; imuYaw[i] = 0;
            imuAccX[i] = 0; imuAccY[i] = 0; imuAccZ[i] = 0;
            imuVeloX[i] = 0; imuVeloY[i] = 0; imuVeloZ[i] = 0;
            imuShiftX[i] = 0; imuShiftY[i] = 0; imuShiftZ[i] = 0;
            imuAngularVeloX[i] = 0; imuAngularVeloY[i] = 0; imuAngularVeloZ[i] = 0;
            imuAngularRotationX[i] = 0; imuAngularRotationY[i] = 0; imuAngularRotationZ[i] = 0;  
        }


        skipFrameNum = 1;

        for (int i = 0; i < 6; ++i){
            transformCur[i] = 0;
            transformSum[i] = 0;
        }

        systemInitedLM = false;

        imuRollLast = 0; imuPitchLast = 0; imuYawLast = 0;
        imuShiftFromStartX = 0; imuShiftFromStartY = 0; imuShiftFromStartZ = 0;
        imuVeloFromStartX = 0; imuVeloFromStartY = 0; imuVeloFromStartZ = 0;

        laserCloudCornerLast.reset(new pcl::PointCloud<PointType>());
        laserCloudSurfLast.reset(new pcl::PointCloud<PointType>());
        laserCloudOri.reset(new pcl::PointCloud<PointType>());
        coeffSel.reset(new pcl::PointCloud<PointType>());

        kdtreeCornerLast.reset(new pcl::KdTreeFLANN<PointType>());
        kdtreeSurfLast.reset(new pcl::KdTreeFLANN<PointType>());

        laserOdometry.header.frame_id = "/camera_init";
        laserOdometry.child_frame_id = "/laser_odom";

        laserOdometryTrans.frame_id_ = "/camera_init";
        laserOdometryTrans.child_frame_id_ = "/laser_odom";
        
        isDegenerate = false;
        matP = cv::Mat(6, 6, CV_32F, cv::Scalar::all(0));  //32位浮点数，单通道，矩阵全用0填充

        frameCount = skipFrameNum;
    }

    // 更新初始时刻i=0时刻的rpy角的正余弦值
    void updateImuRollPitchYawStartSinCos(){
        cosImuRollStart = cos(imuRollStart);
        cosImuPitchStart = cos(imuPitchStart);
        cosImuYawStart = cos(imuYawStart);
        sinImuRollStart = sin(imuRollStart);
        sinImuPitchStart = sin(imuPitchStart);
        sinImuYawStart = sin(imuYawStart);
    }


    void ShiftToStartIMU(float pointTime)
    {   // 下面三个量表示的是世界坐标系下，从start到cur的坐标的漂移(IMU加速度造成的匀速运动位移偏差)
        imuShiftFromStartXCur = imuShiftXCur - imuShiftXStart - imuVeloXStart * pointTime;  //P' = P_cur-P_start-V_start*t
        imuShiftFromStartYCur = imuShiftYCur - imuShiftYStart - imuVeloYStart * pointTime;
        imuShiftFromStartZCur = imuShiftZCur - imuShiftZStart - imuVeloZStart * pointTime
        
        // *从世界坐标系变换到startIMU坐标系（和VeloToStartIMU函数的思路一样)
        //首先绕y轴转(-yaw)的角度
        float x1 = cosImuYawStart * imuShiftFromStartXCur - sinImuYawStart * imuShiftFromStartZCur;
        float y1 = imuShiftFromStartYCur;
        float z1 = sinImuYawStart * imuShiftFromStartXCur + cosImuYawStart * imuShiftFromStartZCur;
        //绕x轴转(-pitch)的角度
        float x2 = x1;
        float y2 = cosImuPitchStart * y1 + sinImuPitchStart * z1;
        float z2 = -sinImuPitchStart * y1 + cosImuPitchStart * z1;
        //绕z轴转(-roll)的角度
        imuShiftFromStartXCur = cosImuRollStart * x2 + sinImuRollStart * y2;
        imuShiftFromStartYCur = -sinImuRollStart * x2 + cosImuRollStart * y2;
        imuShiftFromStartZCur = z2;
    }

    void VeloToStartIMU()
    {
        // imuVeloXStart,imuVeloYStart,imuVeloZStart是点云索引i=0时刻的速度
        //* 此处计算的是相对于每帧初始时刻i=0时的相对速度
        // 这个相对速度在世界坐标系下
        imuVeloFromStartXCur = imuVeloXCur - imuVeloXStart;
        imuVeloFromStartYCur = imuVeloYCur - imuVeloYStart;
        imuVeloFromStartZCur = imuVeloZCur - imuVeloZStart;

        // ！！！下面从世界坐标系转换到StartIMU坐标系，roll,pitch,yaw要取负值 
        //todo (R_start_w = R_w_start.inverse() = Rz(-roll)*Rx(-pitch)*Ry(-yaw),  其中R_w_start = Ry(yaw)*Rx(pitch)*Rz(roll),可以看成反过来进行旋转操作)
        // 首先绕y轴进行旋转（-yaw）的角度        
        //    |cosry   0   sinry|
        // Ry=|0       1       0|
        //    |-sinry  0   cosry|
        float x1 = cosImuYawStart * imuVeloFromStartXCur - sinImuYawStart * imuVeloFromStartZCur;
        float y1 = imuVeloFromStartYCur;
        float z1 = sinImuYawStart * imuVeloFromStartXCur + cosImuYawStart * imuVeloFromStartZCur;

        // 绕当前x轴旋转(-pitch)的角度
        //    |1     0        0|
        // Rx=|0   cosrx -sinrx|
        //    |0   sinrx  cosrx|
        float x2 = x1;
        float y2 = cosImuPitchStart * y1 + sinImuPitchStart * z1;
        float z2 = -sinImuPitchStart * y1 + cosImuPitchStart * z1;

        // 绕当前z轴旋转(-roll)的角度
        //     |cosrz  -sinrz  0|
        //  Rz=|sinrz  cosrz   0|
        //     |0       0      1|
        imuVeloFromStartXCur = cosImuRollStart * x2 + sinImuRollStart * y2;         
        imuVeloFromStartYCur = -sinImuRollStart * x2 + cosImuRollStart * y2;
        imuVeloFromStartZCur = z2;
    }

    //todo 该函数的功能是把点云坐标变换到初始imu时刻
    void TransformToStartIMU(PointType *p)
    {
        // 变换顺序：Cur-->世界坐标系-->Start，这两次变换中，
        // 前一次是正变换，角度为正，后一次是逆变换，角度应该为负
        // 可以参考：
        // https://blog.csdn.net/wykxwyc/article/details/101712524
        //* 1.Cur-->世界坐标系
        // 因为在adjustDistortion函数中有对xyz的坐标进行交换的过程
        // 交换的过程是x=原来的y，y=原来的z，z=原来的x
        // 所以下面其实是绕Z轴(原先的x轴)旋转，对应的是roll角
        //
        //     |cosrz  -sinrz  0|
        //  Rz=|sinrz  cosrz   0|
        //     |0       0      1|
        // [x1,y1,z1]^T=Rz*[x,y,z]
        float x1 = cos(imuRollCur) * p->x - sin(imuRollCur) * p->y;
        float y1 = sin(imuRollCur) * p->x + cos(imuRollCur) * p->y;
        float z1 = p->z;

        // 绕X轴(原先的y轴)旋转,对应为pitch角
        // 
        // [x2,y2,z2]^T=Rx*[x1,y1,z1]
        //    |1     0        0|
        // Rx=|0   cosrx -sinrx|
        //    |0   sinrx  cosrx|
        float x2 = x1;
        float y2 = cos(imuPitchCur) * y1 - sin(imuPitchCur) * z1;
        float z2 = sin(imuPitchCur) * y1 + cos(imuPitchCur) * z1;

        // 最后再绕Y轴(原先的Z轴)旋转，对应为yaw角
        //    |cosry   0   sinry|
        // Ry=|0       1       0|
        //    |-sinry  0   cosry|
        float x3 = cos(imuYawCur) * x2 + sin(imuYawCur) * z2;
        float y3 = y2;
        float z3 = -sin(imuYawCur) * x2 + cos(imuYawCur) * z2;

        //*下面部分的代码功能是从世界坐标系的原点变换到StartIMU坐标系(从世界坐标系变换到startIMU坐标系)
        // 变换方式和函数VeloToStartIMU()中的类似
        //* 2.世界坐标系-->StartIMU
        // 首先绕y轴进行旋转（-yaw）的角度        
        //    |cosry   0   sinry|
        // Ry=|0       1       0|
        //    |-sinry  0   cosry|
        float x4 = cosImuYawStart * x3 - sinImuYawStart * z3;
        float y4 = y3;
        float z4 = sinImuYawStart * x3 + cosImuYawStart * z3;

        // 绕当前x轴旋转(-pitch)的角度
        //    |1     0        0|
        // Rx=|0   cosrx -sinrx|
        //    |0   sinrx  cosrx|
        float x5 = x4;
        float y5 = cosImuPitchStart * y4 + sinImuPitchStart * z4;
        float z5 = -sinImuPitchStart * y4 + cosImuPitchStart * z4;

        // 绕当前z轴旋转(-roll)的角度+坐标系的漂移(StartIMU下的漂移)  （最终激光点转换到匀速运动假设情况下StartIMU坐标系)
        //     |cosrz  -sinrz  0|
        //  Rz=|sinrz  cosrz   0|
        //     |0       0      1|
        p->x = cosImuRollStart * x5 + sinImuRollStart * y5 + imuShiftFromStartXCur;   //ShiftToStartIMU()函数中有漂移计算（加上IMU位置运动补偿）
        p->y = -sinImuRollStart * x5 + cosImuRollStart * y5 + imuShiftFromStartYCur;
        p->z = z5 + imuShiftFromStartZCur;
    }

    // 获取每一帧IMU数据对应IMU在全局坐标系下的位移和速度
    void AccumulateIMUShiftAndRotation()
    {   
        //获得由IMUHandler函数得到该帧IMU数据的欧拉角和三轴角加速度
        float roll = imuRoll[imuPointerLast];      //世界坐标系下的RPY
        float pitch = imuPitch[imuPointerLast];
        float yaw = imuYaw[imuPointerLast];
        float accX = imuAccX[imuPointerLast];   //imu-->右手坐标系的加速值
        float accY = imuAccY[imuPointerLast];
        float accZ = imuAccZ[imuPointerLast];

        //*将当前时刻的加速度值绕交换过的ZXY固定轴（原XYZ）分别旋转(roll, pitch, yaw)角，转换得到世界坐标系  （原：R_w_i = Rz(yaw)* Ry(pitch)*Rx(roll)， 现R_w_c = Ry(yaw)*Rx(pitch)*Rz(roll) )
        // 先绕Z轴(原x轴)旋转,下方坐标系示意imuHandler()中加速度的坐标轴交换
        //  z->Y
        //  ^  
        //  |    ^ y->X
        //  |   /
        //  |  /
        //  | /
        //  -----> x->Z
        //
        //     |cosrz  -sinrz  0|
        //  Rz=|sinrz  cosrz   0|
        //     |0       0      1|
        // [x1,y1,z1]^T=Rz*[accX,accY,accZ]
        float x1 = cos(roll) * accX - sin(roll) * accY;
        float y1 = sin(roll) * accX + cos(roll) * accY;
        float z1 = accZ;

        // 绕X轴(原y轴)旋转
        // [x2,y2,z2]^T=Rx*[x1,y1,z1]
        //    |1     0        0|
        // Rx=|0   cosrx -sinrx|
        //    |0   sinrx  cosrx|
        float x2 = x1;
        float y2 = cos(pitch) * y1 - sin(pitch) * z1;
        float z2 = sin(pitch) * y1 + cos(pitch) * z1;

        // 最后再绕Y轴(原z轴)旋转
        //    |cosry   0   sinry|
        // Ry=|0       1       0|
        //    |-sinry  0   cosry|
        accX = cos(yaw) * x2 + sin(yaw) * z2;
        accY = y2;
        accZ = -sin(yaw) * x2 + cos(yaw) * z2;
         
        //*进行位移，速度，角度量的累加
        int imuPointerBack = (imuPointerLast + imuQueLength - 1) % imuQueLength;  //上一个imu点
        double timeDiff = imuTime[imuPointerLast] - imuTime[imuPointerBack];   //上一个点到当前点所经历的时间，即计算imu测量周期
        //要求imu的频率至少比lidar高，这样的imu信息才使用，后面校正也才有意义
        if (timeDiff < scanPeriod) {    //extern const float scanPeriod = 0.1;
            //*求每个imu时间点的位移与速度,两点之间视为匀加速直线运动  
            //s1 = s2+ vt + 1/2at*t (位移)
            imuShiftX[imuPointerLast] = imuShiftX[imuPointerBack] + imuVeloX[imuPointerBack] * timeDiff + accX * timeDiff * timeDiff / 2;   //世界坐标系下的平移
            imuShiftY[imuPointerLast] = imuShiftY[imuPointerBack] + imuVeloY[imuPointerBack] * timeDiff + accY * timeDiff * timeDiff / 2;
            imuShiftZ[imuPointerLast] = imuShiftZ[imuPointerBack] + imuVeloZ[imuPointerBack] * timeDiff + accZ * timeDiff * timeDiff / 2;
            //v1 = v2+a*t（速度）
            imuVeloX[imuPointerLast] = imuVeloX[imuPointerBack] + accX * timeDiff;  //世界坐标系下的速度
            imuVeloY[imuPointerLast] = imuVeloY[imuPointerBack] + accY * timeDiff;
            imuVeloZ[imuPointerLast] = imuVeloZ[imuPointerBack] + accZ * timeDiff;
            //R1 = R2+wt (角度)
            imuAngularRotationX[imuPointerLast] = imuAngularRotationX[imuPointerBack] + imuAngularVeloX[imuPointerBack] * timeDiff;   //世界坐标系下角度旋转量
            imuAngularRotationY[imuPointerLast] = imuAngularRotationY[imuPointerBack] + imuAngularVeloY[imuPointerBack] * timeDiff;
            imuAngularRotationZ[imuPointerLast] = imuAngularRotationZ[imuPointerBack] + imuAngularVeloZ[imuPointerBack] * timeDiff;
        }
    }

    //!imu的回调函数
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn)    //imu消息格式链接：http://docs.ros.org/en/api/sensor_msgs/html/msg/Imu.html
    {  
        double roll, pitch, yaw;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(imuIn->orientation, orientation);   //将ros消息格式中的姿态转成tf的格式
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);   //然后将四元数转成欧拉角,该值为世界坐标系下的值

        //1.求出世界坐标系到IMU坐标系的旋转矩阵，即R_i_w = R_w_i.inverse()   ( note: R_w_i = Rz(yaw)* Ry(pitch)*Rx(roll)  )
        //2.imu每个轴分别减去重力加速度的影响 Vi = R_i_w * Vg（note:重力方向和imu所获数值方向相反,所以数值为 Vg=[0,0,-9.8]）
        //3.将IMU坐标系转换到右手坐标系  (note：变换之后旋转矩阵 R = Ry(yaw)*Rx(pitch)*Rz(roll))
            //hand<----imu
            //     x  =  y     
            //     y  =  z     
            //     z  =  x     
        //*加速度去除重力影响，同时坐标轴进行变换 (参考文章链接：https://blog.csdn.net/jtcap/article/details/123855216)
        float accX = imuIn->linear_acceleration.y - sin(roll) * cos(pitch) * 9.81;
        float accY = imuIn->linear_acceleration.z - cos(roll) * cos(pitch) * 9.81;
        float accZ = imuIn->linear_acceleration.x + sin(pitch) * 9.81;

        //循环移位效果，形成环形数组
        imuPointerLast = (imuPointerLast + 1) % imuQueLength;   //imuPointerLast初始值为-1,imu回调调用一次，imuPointerLast增1（note:这是个循环队列，不是无限增加)

        imuTime[imuPointerLast] = imuIn->header.stamp.toSec();  //获取IMU时间戳信息(结果为秒)
        //  imu姿态获取的RPY
        imuRoll[imuPointerLast] = roll;   //世界坐标系的角度
        imuPitch[imuPointerLast] = pitch;
        imuYaw[imuPointerLast] = yaw;
        //imu的加速度
        imuAccX[imuPointerLast] = accX;  //值为imu-->右手坐标系的加速度值
        imuAccY[imuPointerLast] = accY;
        imuAccZ[imuPointerLast] = accZ;
        //imu的角速度
        imuAngularVeloX[imuPointerLast] = imuIn->angular_velocity.x;   //世界坐标系下的角速度
        imuAngularVeloY[imuPointerLast] = imuIn->angular_velocity.y;
        imuAngularVeloZ[imuPointerLast] = imuIn->angular_velocity.z;

        AccumulateIMUShiftAndRotation();
    }

    //!点云回调函数(分割的点云)
    void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        cloudHeader = laserCloudMsg->header;  //点云的头信息

        timeScanCur = cloudHeader.stamp.toSec();  //将时间戳转成秒
        timeNewSegmentedCloud = timeScanCur;

        segmentedCloud->clear();
        pcl::fromROSMsg(*laserCloudMsg, *segmentedCloud);  //将ros消息格式转成点云格式

        newSegmentedCloud = true;  //初始值为false
    }

    //!异常点云的回调函数
    void outlierCloudHandler(const sensor_msgs::PointCloud2ConstPtr& msgIn){

        timeNewOutlierCloud = msgIn->header.stamp.toSec();  //取出时间信息(时间戳转成秒)

        outlierCloud->clear();
        pcl::fromROSMsg(*msgIn, *outlierCloud);  //ros消息格式转成点云格式

        newOutlierCloud = true;
    }

    //!信息点云的回调函数
    void laserCloudInfoHandler(const cloud_msgs::cloud_infoConstPtr& msgIn)
    {
        timeNewSegmentedCloudInfo = msgIn->header.stamp.toSec();  //时间戳转成秒
        segInfo = *msgIn;
        newSegmentedCloudInfo = true;
    }

    void adjustDistortion()
    {
        bool halfPassed = false;
        int cloudSize = segmentedCloud->points.size();  //取出一帧分割云的大小

        PointType point;

        for (int i = 0; i < cloudSize; i++) {
            // 这里xyz与laboshin_loam代码中的一样经过坐标轴变换
            // imuhandler() 中相同的坐标变换
            //*将激光雷达坐标系转到右手坐标系
            //hand <-------------- lidar
            //   x  =   y
            //   y  =   z
            //   z  =   x
            point.x = segmentedCloud->points[i].y;          
            point.y = segmentedCloud->points[i].z;
            point.z = segmentedCloud->points[i].x;

            // -atan2(p.x,p.z)==>-atan2(y,x)
            // ori表示的是偏航角yaw，因为前面有负号，ori=[-M_PI,M_PI)
            // 因为segInfo.orientationDiff表示的范围是(PI,3PI)，在2PI附近
            //*下面过程的主要作用是调整ori大小，满足start<ori<end
            float ori = -atan2(point.x, point.z);
            if (!halfPassed) {  //旋转未过半，与起始角度进行比较
                if (ori < segInfo.startOrientation - M_PI / 2)  // start-ori>M_PI/2，说明ori小于start，不合理，正常情况在前半圈的话，ori-stat范围[0,M_PI]
                    ori += 2 * M_PI;
                else if (ori > segInfo.startOrientation + M_PI * 3 / 2)  // ori-start>3/2*M_PI,说明ori太大，不合理
                    ori -= 2 * M_PI;

                if (ori - segInfo.startOrientation > M_PI)
                    halfPassed = true;
            } else {    //旋转过半，与结束角度进行比较
                ori += 2 * M_PI;

                if (ori < segInfo.endOrientation - M_PI * 3 / 2)  // end-ori>3/2*PI,ori太小
                    ori += 2 * M_PI;
                else if (ori > segInfo.endOrientation + M_PI / 2)   // ori-end>M_PI/2,太大
                    ori -= 2 * M_PI;
            }

            // 用 point.intensity 来保存时间
            float relTime = (ori - segInfo.startOrientation) / segInfo.orientationDiff;  //角度差的比例，和A-LOAM中一样
            point.intensity = int(segmentedCloud->points[i].intensity) + scanPeriod * relTime;   //extern const float scanPeriod = 0.1;  点的强度值 = 线号+相对时间

            if (imuPointerLast >= 0) {
                float pointTime = relTime * scanPeriod;
                imuPointerFront = imuPointerLastIteration;   //imuPointerLastIteration初始值为0,然后imuPointerLastIteration = imuPointerLast;
                // while循环内进行时间轴对齐
                while (imuPointerFront != imuPointerLast) {
                    //寻找是否有点云的时间戳小于IMU的时间戳的IMU位置:imuPointerFront
                    if (timeScanCur + pointTime < imuTime[imuPointerFront]) {   //timeScanCur为当前帧起始点云的时间戳，imuTime数组中存放的是imu的时间戳
                        break;
                    }
                    imuPointerFront = (imuPointerFront + 1) % imuQueLength;
                }

                //没找到,此时imuPointerFront==imtPointerLast,只能以当前收到的最新的IMU的速度，位移，欧拉角作为当前点的速度，位移，欧拉角使用
                if (timeScanCur + pointTime > imuTime[imuPointerFront]) {  
                    // 该条件内imu数据比激光数据早，但是没有更后面的数据
                    // (打个比方,激光在9点时出现，imu现在只有8点的)
                    // 这种情况上面while循环是以imuPointerFront == imuPointerLast结束的
                    imuRollCur = imuRoll[imuPointerFront];  //世界坐标系角度
                    imuPitchCur = imuPitch[imuPointerFront];
                    imuYawCur = imuYaw[imuPointerFront];

                    imuVeloXCur = imuVeloX[imuPointerFront];  //世界坐标系下的速度
                    imuVeloYCur = imuVeloY[imuPointerFront];
                    imuVeloZCur = imuVeloZ[imuPointerFront];

                    imuShiftXCur = imuShiftX[imuPointerFront];  //世界坐标系下的位移
                    imuShiftYCur = imuShiftY[imuPointerFront];
                    imuShiftZCur = imuShiftZ[imuPointerFront];   
                } else {//*找到了点云时间戳小于IMU时间戳的IMU位置,则该点必处于imuPointerBack和imuPointerFront之间，据此线性插值，计算点云点的速度，位移和欧拉角
                    // 在imu数据充足的情况下可以进行插补
                    // 当前timeScanCur + pointTime < imuTime[imuPointerFront]，
                    // 而且imuPointerFront是最早一个时间大于timeScanCur + pointTime的imu数据指针
                    int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
                    //按时间距离计算权重分配比率,用于后续线性插值
                    float ratioFront = (timeScanCur + pointTime - imuTime[imuPointerBack]) 
                                                     / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
                    float ratioBack = (imuTime[imuPointerFront] - timeScanCur - pointTime) 
                                                    / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
                    // 通过上面计算的ratioFront以及ratioBack进行插补
                    // 因为imuRollCur和imuPitchCur通常都在0度左右，变化不会很大，因此不需要考虑超过2M_PI的情况
                    imuRollCur = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;        //*线性插值 y = y2{(x-x1)/(x2-x1)} + y1{(x2-x)/(x2-x1)}
                    imuPitchCur = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;

                    // imuYaw转的角度比较大，需要考虑超过2*M_PI的情况
                    if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] > M_PI) {
                        imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] + 2 * M_PI) * ratioBack;
                    } else if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] < -M_PI) {
                        imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] - 2 * M_PI) * ratioBack;
                    } else {
                        imuYawCur = imuYaw[imuPointerFront] * ratioFront + imuYaw[imuPointerBack] * ratioBack;
                    }

                    // imu速度插补
                    imuVeloXCur = imuVeloX[imuPointerFront] * ratioFront + imuVeloX[imuPointerBack] * ratioBack;
                    imuVeloYCur = imuVeloY[imuPointerFront] * ratioFront + imuVeloY[imuPointerBack] * ratioBack;
                    imuVeloZCur = imuVeloZ[imuPointerFront] * ratioFront + imuVeloZ[imuPointerBack] * ratioBack;

                    // imu位移插补
                    imuShiftXCur = imuShiftX[imuPointerFront] * ratioFront + imuShiftX[imuPointerBack] * ratioBack;
                    imuShiftYCur = imuShiftY[imuPointerFront] * ratioFront + imuShiftY[imuPointerBack] * ratioBack;
                    imuShiftZCur = imuShiftZ[imuPointerFront] * ratioFront + imuShiftZ[imuPointerBack] * ratioBack;
                }

                //*目的：点云插补到startIMU坐标系
                if (i == 0) {  //如果是第一个点,记住点云起始位置的速度，位移，欧拉角
                    // 此处更新过的角度值主要用在updateImuRollPitchYawStartSinCos()中,
                    // 更新每个角的正余弦值
                    imuRollStart = imuRollCur;
                    imuPitchStart = imuPitchCur;
                    imuYawStart = imuYawCur;

                    imuVeloXStart = imuVeloXCur;
                    imuVeloYStart = imuVeloYCur;
                    imuVeloZStart = imuVeloZCur;

                    imuShiftXStart = imuShiftXCur;
                    imuShiftYStart = imuShiftYCur;
                    imuShiftZStart = imuShiftZCur;

                    if (timeScanCur + pointTime > imuTime[imuPointerFront]) {  // 该条件内imu数据比激光数据早，但是没有更后面的数据
                        imuAngularRotationXCur = imuAngularRotationX[imuPointerFront];
                        imuAngularRotationYCur = imuAngularRotationY[imuPointerFront];
                        imuAngularRotationZCur = imuAngularRotationZ[imuPointerFront];  //世界坐标系下的值
                    }else{
                        // 在imu数据充足的情况下可以进行插补
                        int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
                        float ratioFront = (timeScanCur + pointTime - imuTime[imuPointerBack]) 
                                                         / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
                        float ratioBack = (imuTime[imuPointerFront] - timeScanCur - pointTime) 
                                                        / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
                        imuAngularRotationXCur = imuAngularRotationX[imuPointerFront] * ratioFront + imuAngularRotationX[imuPointerBack] * ratioBack;
                        imuAngularRotationYCur = imuAngularRotationY[imuPointerFront] * ratioFront + imuAngularRotationY[imuPointerBack] * ratioBack;
                        imuAngularRotationZCur = imuAngularRotationZ[imuPointerFront] * ratioFront + imuAngularRotationZ[imuPointerBack] * ratioBack;
                    }
                    // 距离上一次插补，旋转过的角度变化值（世界坐标系下的值)（两帧之间起始位置相差的角度量)
                    imuAngularFromStartX = imuAngularRotationXCur - imuAngularRotationXLast;
                    imuAngularFromStartY = imuAngularRotationYCur - imuAngularRotationYLast;
                    imuAngularFromStartZ = imuAngularRotationZCur - imuAngularRotationZLast;
                  
                    imuAngularRotationXLast = imuAngularRotationXCur;
                    imuAngularRotationYLast = imuAngularRotationYCur;
                    imuAngularRotationZLast = imuAngularRotationZCur;

                    // 这里更新的是i=0时刻的rpy角，后面将速度坐标投影过来会用到i=0时刻的值
                    updateImuRollPitchYawStartSinCos();
                } else {
                    // 速度投影到初始i=0时刻
                    VeloToStartIMU();
                    // 将点的坐标变换到初始i=0时刻
                    TransformToStartIMU(&point);
                }
            }
            segmentedCloud->points[i] = point;           //*点发生了变化，之前是右手坐标系--->StartIMU坐标系，且其强度值 = 线号+相对时间  (最终激光点转换到匀速运动假设情况下StartIMU坐标系)
        }

        imuPointerLastIteration = imuPointerLast;     
    }

    // 计算光滑性，这里的计算没有完全按照公式进行，
    // 缺少除以总点数i和r[i]
    void calculateSmoothness()   
    {
        int cloudSize = segmentedCloud->points.size();
        for (int i = 5; i < cloudSize - 5; i++) {
            //光滑性的本质：判断点的周围点，与点之间的差距，差距越大，光滑性越差，反之光滑性越好
            float diffRange = segInfo.segmentedCloudRange[i-5] + segInfo.segmentedCloudRange[i-4]
                            + segInfo.segmentedCloudRange[i-3] + segInfo.segmentedCloudRange[i-2]
                            + segInfo.segmentedCloudRange[i-1] - segInfo.segmentedCloudRange[i] * 10
                            + segInfo.segmentedCloudRange[i+1] + segInfo.segmentedCloudRange[i+2]
                            + segInfo.segmentedCloudRange[i+3] + segInfo.segmentedCloudRange[i+4]
                            + segInfo.segmentedCloudRange[i+5];            

            cloudCurvature[i] = diffRange*diffRange;
            //在markOccludedPoints()函数中对该参数进行重新修改
            cloudNeighborPicked[i] = 0;
            // 在extractFeatures()函数中会对标签进行修改，
			// 初始化为0，surfPointsFlat标记为-1，surfPointsLessFlatScan为不大于0的标签
			// cornerPointsSharp标记为2，cornerPointsLessSharp标记为1
            cloudLabel[i] = 0;

            cloudSmoothness[i].value = cloudCurvature[i];
            cloudSmoothness[i].ind = i;
        }
    }

    //标记一下遮挡的点
    void markOccludedPoints()
    {
        int cloudSize = segmentedCloud->points.size();
        //标记遮挡的点和平行线束的点
        for (int i = 5; i < cloudSize - 6; ++i){
            //取出相邻两个点的距离信息
            float depth1 = segInfo.segmentedCloudRange[i];      //取出点的距离
            float depth2 = segInfo.segmentedCloudRange[i+1];
            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[i+1] - segInfo.segmentedCloudColInd[i]));  //计算两个点之间的列索引之差

            if (columnDiff < 10){
                /*
                    两个点在同一扫描线上，且距离相差大于0.3，认为存在遮挡关系（也就是这两个点不在同一平面上，如果在同一平面上，距离相差不会太大）
                    远处的点会被遮挡，标记一下该点以及相邻的5个点，后面不再进行特征提取
                */
                if (depth1 - depth2 > 0.3){                      //depth1远被遮挡，因此其之前的5个点都设置为无效点
                    cloudNeighborPicked[i - 5] = 1;
                    cloudNeighborPicked[i - 4] = 1;
                    cloudNeighborPicked[i - 3] = 1;
                    cloudNeighborPicked[i - 2] = 1;
                    cloudNeighborPicked[i - 1] = 1;
                    cloudNeighborPicked[i] = 1;
                }else if (depth2 - depth1 > 0.3){            //同理
                    cloudNeighborPicked[i + 1] = 1;
                    cloudNeighborPicked[i + 2] = 1;
                    cloudNeighborPicked[i + 3] = 1;
                    cloudNeighborPicked[i + 4] = 1;
                    cloudNeighborPicked[i + 5] = 1;
                    cloudNeighborPicked[i + 6] = 1;
                }
            }
            // parallel beam
            //用前后相邻点判断当前点所在平面是否与激光束方向平行（如果两点距离比较大 就很可能是平行的点，也很可能失去观测）
            float diff1 = std::abs(float(segInfo.segmentedCloudRange[i-1] - segInfo.segmentedCloudRange[i]));  //前一个点与当前点之间的距离
            float diff2 = std::abs(float(segInfo.segmentedCloudRange[i+1] - segInfo.segmentedCloudRange[i])); //后一个点与当前点之间的距离
            //如果当前点距离左右邻点都过远，则视其为瑕点，因为入射角可能太小导致误差较大
            if (diff1 > 0.02 * segInfo.segmentedCloudRange[i] && diff2 > 0.02 * segInfo.segmentedCloudRange[i])
                cloudNeighborPicked[i] = 1;
        }
    }
    
    void extractFeatures()
    {
        //指针清空
        cornerPointsSharp->clear();
        cornerPointsLessSharp->clear();
        surfPointsFlat->clear();
        surfPointsLessFlat->clear();

        for (int i = 0; i < N_SCAN; i++) {  //遍历每一根扫描线

            surfPointsLessFlatScan->clear();

            for (int j = 0; j < 6; j++) {  // 将一条扫描线扫描一周的点云数据，划分为6段，每段分开提取有限数量的特征，保证特征均匀分布
                /*
                 根据之前得到的每个scan的起始和结束id来均分：（startRingIndex和 endRingIndex 在imageProjection.cpp中的 cloudExtraction函数里被填入）
                    startRingIndex为扫描线起始第5个激光点在一维数组中的索引（注意：所有的点云在这里都是以"一维数组"的形式保存）   
                    
                    !假设 当前ring在一维数组中起始点是m，结尾点为n（不包括n），那么6段的起始点分别为：
                    *m + [(n-m)/6]*j  （ j从0～5） 
                    *化简为 [（6-j)*m + nj ]/6
                    !6段的终止点分别为：
                    *m + [(n-m)/6]*j+ (n-m)/6  -1           （每段终=每段起+1/6均分段-1）（ j从0～5）(减去1,避免每段首尾重复)
                    *化简为 [（5-j)*m + (j+1)*n ]/6 -1
                    这块不必细究边缘值到底是不是划分的准（例如考虑前五个点是不是都不要，还是说只不要前四个点），
                    只是尽可能的分开成六段，首位相接的地方不要。因为庞大的点云中，一两个点其实无关紧要
                */
                int sp = (segInfo.startRingIndex[i] * (6 - j)    + segInfo.endRingIndex[i] * j) / 6;
                int ep = (segInfo.startRingIndex[i] * (5 - j)    + segInfo.endRingIndex[i] * (j + 1)) / 6 - 1;

                // 这种情况就不正常（起始>结束）
                if (sp >= ep)
                    continue;

                std::sort(cloudSmoothness.begin()+sp, cloudSmoothness.begin()+ep, by_value());   //按照曲率从小到大升序排列
                //todo 提取角点(2个极大边线点，次极大边线点为20(包含2个极大边线点))
                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--) {  //按照曲率从大到小遍历每段
                    int ind = cloudSmoothness[k].ind;  // 找到这个点对应的原先的idx
                    // 如果没有被认为是遮挡,或平行点且曲率大于边缘点门限（0.1)
                    if (cloudNeighborPicked[ind] == 0 &&     //cloudNeighborPicked中初始值为0，
                        cloudCurvature[ind] > edgeThreshold &&   //const float edgeThreshold = 0.1;
                        segInfo.segmentedCloudGroundFlag[ind] == false) {  //初始值为false,在imageProjection中定义(note: true是地面点，false其他点)
                    
                        largestPickedNum++;  //角点计数
                        // 每段最多找20个角点（ 每段只取20个角点，如果单条扫描线扫描一周是1800个点，则划分6段，每段300个点，从中提取20个角点）
                        //*2个极大边线点，次极大边线点为20(包含2个极大边线点)
                        if (largestPickedNum <= 2) {  
                            // 论文中nFe=2,cloudSmoothness已经按照从小到大的顺序排列，
                            // 所以这边只要选择最后两个放进队列即可
                            // cornerPointsSharp标记为2
                            cloudLabel[ind] = 2;
                            cornerPointsSharp->push_back(segmentedCloud->points[ind]);
                            cornerPointsLessSharp->push_back(segmentedCloud->points[ind]);
                        } else if (largestPickedNum <= 20) {
                            // 塞20个点到cornerPointsLessSharp中去
							// cornerPointsLessSharp标记为1
                            cloudLabel[ind] = 1;
                            cornerPointsLessSharp->push_back(segmentedCloud->points[ind]);
                        } else {
                            break;
                        }
                        
                        cloudNeighborPicked[ind] = 1;  // 标记已被处理
                        // 将这个点周围的几个点（10个点）设置成遮挡点，避免选取太集中
                        for (int l = 1; l <= 5; l++) {   // 同一条扫描线上后5个点标记一下，不再处理，避免特征聚集
                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l - 1]));
                            if (columnDiff > 10)  // 列idx距离太远就算了，空间上也不会太集中
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {  //同理（同一条扫描线上前5个点标记一下，不再处理，避免特征聚集）
                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }
                //todo 提取面点
                int smallestPickedNum = 0;
                for (int k = sp; k <= ep; k++) {  //曲率从小到大遍历每段
                    int ind = cloudSmoothness[k].ind;
                    if (cloudNeighborPicked[ind] == 0 &&                                    // 同样要求不是遮挡点或平行点且曲率小于给定阈值0.1
                        cloudCurvature[ind] < surfThreshold &&
                        segInfo.segmentedCloudGroundFlag[ind] == true) {   //true为地面点，即从地面点中提取面特征

                        cloudLabel[ind] = -1;     //极小平面点标签为-1（note:初始值为0)
                        surfPointsFlat->push_back(segmentedCloud->points[ind]);

                        smallestPickedNum++;
                        if (smallestPickedNum >= 4) {     // 论文中nFp=4，将4个最平的平面点放入队列中
                            break;
                        }

                        cloudNeighborPicked[ind] = 1;  // 标记已被处理
                        // 将这个点周围的几个点（10个点）进行标记，避免特征选取太集中
                        for (int l = 1; l <= 5; l++) {
                            // 从前面往后判断是否是需要的邻接点，是的话就进行标记
                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l - 1]));
                            if (columnDiff > 10)     //相邻点列索引差距太大，跳过不处理
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--) {
                            // 从后往前开始标记
                            int columnDiff = std::abs(int(segInfo.segmentedCloudColInd[ind + l] - segInfo.segmentedCloudColInd[ind + l + 1]));
                            if (columnDiff > 10)
                                break;
                            cloudNeighborPicked[ind + l] = 1;
                        }
                    }
                }

                for (int k = sp; k <= ep; k++) {  // 注意这里是小于等于0,也就是说不是角点的都认为是次极小平面点了(包含4个极小平面点)（cloudLabel初始值为0）
                    if (cloudLabel[k] <= 0) {
                        surfPointsLessFlatScan->push_back(segmentedCloud->points[k]);
                    }
                }
            }
            // 因为面点太多了，所以做一个下采样，减少计算量（note: 下采样滤波器设置叶子大小 0.2*0.2*0.2）
            surfPointsLessFlatScanDS->clear();
            downSizeFilter.setInputCloud(surfPointsLessFlatScan);
            downSizeFilter.filter(*surfPointsLessFlatScanDS);
            // 加入平面点云集合
            *surfPointsLessFlat += *surfPointsLessFlatScanDS;
        }
    }

    void publishCloud()  //发布四种点云(极大边线点+次极大边线点+平面点+次极小平面点)
    {
        sensor_msgs::PointCloud2 laserCloudOutMsg;

	    if (pubCornerPointsSharp.getNumSubscribers() != 0){
	        pcl::toROSMsg(*cornerPointsSharp, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = "/camera";
	        pubCornerPointsSharp.publish(laserCloudOutMsg);
	    }

	    if (pubCornerPointsLessSharp.getNumSubscribers() != 0){
	        pcl::toROSMsg(*cornerPointsLessSharp, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = "/camera";
	        pubCornerPointsLessSharp.publish(laserCloudOutMsg);
	    }

	    if (pubSurfPointsFlat.getNumSubscribers() != 0){
	        pcl::toROSMsg(*surfPointsFlat, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = "/camera";
	        pubSurfPointsFlat.publish(laserCloudOutMsg);
	    }

	    if (pubSurfPointsLessFlat.getNumSubscribers() != 0){
	        pcl::toROSMsg(*surfPointsLessFlat, laserCloudOutMsg);
	        laserCloudOutMsg.header.stamp = cloudHeader.stamp;
	        laserCloudOutMsg.header.frame_id = "/camera";
	        pubSurfPointsLessFlat.publish(laserCloudOutMsg);
	    }
    }







































/*****************************************************************************
    将当前帧点云TransformToStart和将上一帧点云TransformToEnd的作用：
         去除畸变，并将两帧点云数据统一到同一个坐标系下计算
*****************************************************************************/
    //当前点云中的点相对第一个点去除因匀速运动产生的畸变，效果相当于得到在点云扫描开始位置静止扫描得到的点云
    void TransformToStart(PointType const * const pi, PointType * const po)
    {
        // intensity代表的是：整数部分ring序号，小数部分是当前点在这一圈中所花的时间
        // 关于intensity， 参考 adjustDistortion() 函数中的定义
        // s代表的其实是一个比例，s的计算方法应该如下：
        // s=(pi->intensity - int(pi->intensity))/SCAN_PERIOD
        // ===> SCAN_PERIOD=0.1(雷达频率为10hz)
        // 以上理解感谢github用户StefanGlaser
        // https://github.com/laboshinl/loam_velodyne/issues/29
        float s = 10 * (pi->intensity - int(pi->intensity));  //s代表比例和A-LOAM中一样

        //*线性插值：根据每个点在点云中的相对位置关系，乘以相应的旋转平移系数，得到该帧初始点坐标系相对于当前激光点坐标系对应关系
        //todo (利用匀速运动假设， T_next_cur = T_last_cur)
        float rx = s * transformCur[0];  //transformCur中存放的是两帧之间的角度变化量和位移量（上一帧->当前帧,即T_t_t-1)
        float ry = s * transformCur[1];
        float rz = s * transformCur[2];
        float tx = s * transformCur[3];
        float ty = s * transformCur[4];
        float tz = s * transformCur[5];
        
        //todo 转到当前帧起始位置( 计算特征点对应于开始时刻坐标系的坐标，先平移在旋转）（个人理解：转到当前帧的起始时刻,即上一帧的结束时刻）
        //本质 P_k  = R_k_k+1 * (P_k+1  -   t_k)
        //! note:此处是以当前点作为起始时刻,所以角度要取负
        //绕z轴(原先x轴)旋转（-rz)
        float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);     
        float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
        float z1 = (pi->z - tz);
        //绕x轴(原先y轴)旋转(-rx)
        float x2 = x1;
        float y2 = cos(rx) * y1 + sin(rx) * z1;
        float z2 = -sin(rx) * y1 + cos(rx) * z1;
        //绕y轴(原先z轴)旋转(-ry)
        po->x = cos(ry) * x2 - sin(ry) * z2;
        po->y = y2;
        po->z = sin(ry) * x2 + cos(ry) * z2;
        po->intensity = pi->intensity;
    }

    // 先转到start，再从start旋转到end
    void TransformToEnd(PointType const * const pi, PointType * const po)
    {   
        float s = 10 * (pi->intensity - int(pi->intensity));

        float rx = s * transformCur[0];
        float ry = s * transformCur[1];
        float rz = s * transformCur[2];
        float tx = s * transformCur[3];
        float ty = s * transformCur[4];
        float tz = s * transformCur[5];

        //todo 转到当前帧起始坐标系
        //平移后绕z轴旋转（-rz）
        float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);
        float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
        float z1 = (pi->z - tz);

        //绕x轴旋转（-rx）
        float x2 = x1;
        float y2 = cos(rx) * y1 + sin(rx) * z1;
        float z2 = -sin(rx) * y1 + cos(rx) * z1;

        //绕y轴旋转（-ry）
        float x3 = cos(ry) * x2 - sin(ry) * z2;
        float y3 = y2;
        float z3 = sin(ry) * x2 + cos(ry) * z2;  //求出了相对于起始点校正的坐标

        //todo 整体向后补偿，转到当前帧结束时刻（补偿到结束时刻, P_end  =  R_end_start * P_start)
        rx = transformCur[0];
        ry = transformCur[1];
        rz = transformCur[2];
        tx = transformCur[3];
        ty = transformCur[4];
        tz = transformCur[5];

        //R_end_start = Rz(rz)*Rx(rx)*Ry(ry)
        //绕y轴旋转（ry）
        float x4 = cos(ry) * x3 + sin(ry) * z3;
        float y4 = y3;
        float z4 = -sin(ry) * x3 + cos(ry) * z3;

        //绕x轴旋转（rx）
        float x5 = x4;
        float y5 = cos(rx) * y4 - sin(rx) * z4;
        float z5 = sin(rx) * y4 + cos(rx) * z4;

        //绕z轴旋转（rz），再平移
        float x6 = cos(rz) * x5 - sin(rz) * y5 + tx;
        float y6 = sin(rz) * x5 + cos(rz) * y5 + ty;
        float z6 = z5 + tz;


        //*将点云转到imuEnd坐标系下
        //todo 减去该帧中非匀速运动造成的漂移，然后转到世界坐标系下
        //平移后绕z轴旋转（imuRollStart）
        float x7 = cosImuRollStart * (x6 - imuShiftFromStartX) 
                 - sinImuRollStart * (y6 - imuShiftFromStartY);
        float y7 = sinImuRollStart * (x6 - imuShiftFromStartX) 
                 + cosImuRollStart * (y6 - imuShiftFromStartY);
        float z7 = z6 - imuShiftFromStartZ;

        //绕x轴旋转（imuPitchStart）
        float x8 = x7;
        float y8 = cosImuPitchStart * y7 - sinImuPitchStart * z7;
        float z8 = sinImuPitchStart * y7 + cosImuPitchStart * z7;

        //绕y轴旋转（imuYawStart）
        float x9 = cosImuYawStart * x8 + sinImuYawStart * z8;
        float y9 = y8;
        float z9 = -sinImuYawStart * x8 + cosImuYawStart * z8;
        
        //todo 从世界坐标系---->imuEnd（imu最后坐标系）
        //绕y轴旋转（-imuYawLast）
        float x10 = cos(imuYawLast) * x9 - sin(imuYawLast) * z9;
        float y10 = y9;
        float z10 = sin(imuYawLast) * x9 + cos(imuYawLast) * z9;

        //绕x轴旋转（-imuPitchLast）
        float x11 = x10;
        float y11 = cos(imuPitchLast) * y10 + sin(imuPitchLast) * z10;
        float z11 = -sin(imuPitchLast) * y10 + cos(imuPitchLast) * z10;

        //绕z轴旋转（-imuRollLast）
        po->x = cos(imuRollLast) * x11 + sin(imuRollLast) * y11;
        po->y = -sin(imuRollLast) * x11 + cos(imuRollLast) * y11;
        po->z = z11;
        //只保留线号
        po->intensity = int(pi->intensity);
    }

    // (rx, ry, rz, imuPitchStart, imuYawStart, imuRollStart, 
    //  imuPitchLast, imuYawLast, imuRollLast, rx, ry, rz)
    void PluginIMURotation(float bcx, float bcy, float bcz, float blx, float bly, float blz, 
                           float alx, float aly, float alz, float &acx, float &acy, float &acz)
    {
        // bcx,bcy,bcz (rx, ry, rz)构成了 R_(k+1)^(0)
        // blx,bly,blz（imuPitchStart, imuYawStart, imuRollStart） 构成了 R_(YXZ)^(-imuStart)
        // alx,aly,alz（imuPitchLast, imuYawLast, imuRollLast）构成了 R_(ZXY)^(imuEnd)

        //每个角度的正余弦值
        float sbcx = sin(bcx);
        float cbcx = cos(bcx);
        float sbcy = sin(bcy);
        float cbcy = cos(bcy);
        float sbcz = sin(bcz);
        float cbcz = cos(bcz);

        float sblx = sin(blx);
        float cblx = cos(blx);
        float sbly = sin(bly);
        float cbly = cos(bly);
        float sblz = sin(blz);
        float cblz = cos(blz);

        float salx = sin(alx);
        float calx = cos(alx);
        float saly = sin(aly);
        float caly = cos(aly);
        float salz = sin(alz);
        float calz = cos(alz);

        //R_0_imuEnd  = R_0_k+1(imuStart)*R_w_imuStart.inverse() * R_w_imuEnd= Ry(bcy)*Rx(bcx)*Rz(bcz)  *  Rz(-blz)*Rx(-blx)*Ry(-bly)  * Ry(aly)*Rx(alx)*Rz(alz)
        //R_0_imuEnd = Ry(ry)*Rx(rx)*Rz(rz)
        float srx = -sbcx*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly) 
                  - cbcx*cbcz*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
                  - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
                  - cbcx*sbcz*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
                  - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz);
        acx = -asin(srx);

        float srycrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
                     - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
                     - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
                     - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz) 
                     + cbcx*sbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
        float crycrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
                     - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz) 
                     - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
                     - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
                     + cbcx*cbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
        acy = atan2(srycrx / cos(acx), crycrx / cos(acx));
        
        float srzcrx = sbcx*(cblx*cbly*(calz*saly - caly*salx*salz) 
                     - cblx*sbly*(caly*calz + salx*saly*salz) + calx*salz*sblx) 
                     - cbcx*cbcz*((caly*calz + salx*saly*salz)*(cbly*sblz - cblz*sblx*sbly) 
                     + (calz*saly - caly*salx*salz)*(sbly*sblz + cbly*cblz*sblx) 
                     - calx*cblx*cblz*salz) + cbcx*sbcz*((caly*calz + salx*saly*salz)*(cbly*cblz 
                     + sblx*sbly*sblz) + (calz*saly - caly*salx*salz)*(cblz*sbly - cbly*sblx*sblz) 
                     + calx*cblx*salz*sblz);
        float crzcrx = sbcx*(cblx*sbly*(caly*salz - calz*salx*saly) 
                     - cblx*cbly*(saly*salz + caly*calz*salx) + calx*calz*sblx) 
                     + cbcx*cbcz*((saly*salz + caly*calz*salx)*(sbly*sblz + cbly*cblz*sblx) 
                     + (caly*salz - calz*salx*saly)*(cbly*sblz - cblz*sblx*sbly) 
                     + calx*calz*cblx*cblz) - cbcx*sbcz*((saly*salz + caly*calz*salx)*(cblz*sbly 
                     - cbly*sblx*sblz) + (caly*salz - calz*salx*saly)*(cbly*cblz + sblx*sbly*sblz) 
                     - calx*calz*cblx*sblz);
        acz = atan2(srzcrx / cos(acx), crzcrx / cos(acx));
    }

    //相对于第一个点云即原点，积累旋转量
    void AccumulateRotation(float cx, float cy, float cz, float lx, float ly, float lz, 
                            float &ox, float &oy, float &oz)
    {   
        //todo R_w_cur = R_w_last * R_curr_last.inverse()= {Ry(cy)*Rx(cx)*Rz(cz) }  *  {Ry(ly)*Rx(lx)*Rz(lz)}  = Ry(oz)*Rx(oy)*Rz(ox)
        //* R_last_cur = Ry(-yaw)*Rx(-pitch)*Rz(-roll) = Ry(ly)*Rx(lx)*Rz(lz)    (以当前点为起始时刻,和TransformToStart()函数中求解思路一致)     //transformCur(当前时刻的相关值): pitch yaw roll x y z         
        //* R_w_last = Ry(yaw)*Rx(pitch)*Rz(roll)=Ry(cy)*Rx(cx)*Rz(cz)     (和TransformToStartIMU()中转到世界坐标系思路一致)  //transformSum(世界坐标系下的值)：pitch yaw roll x y z                                       
        //R_w_cur矩阵的R12（第二行第三列对应的元素)
        float srx = cos(lx)*cos(cx)*sin(ly)*sin(cz) - cos(cx)*cos(cz)*sin(lx) - cos(lx)*cos(ly)*sin(cx);
        ox = -asin(srx);

        //R_w_cur矩阵的R02（第1行第3列对应的元素)
        float srycrx = sin(lx)*(cos(cy)*sin(cz) - cos(cz)*sin(cx)*sin(cy)) + cos(lx)*sin(ly)*(cos(cy)*cos(cz) 
                     + sin(cx)*sin(cy)*sin(cz)) + cos(lx)*cos(ly)*cos(cx)*sin(cy);
        //R_w_cur矩阵的R22（第3行第3列对应的元素)
        float crycrx = cos(lx)*cos(ly)*cos(cx)*cos(cy) - cos(lx)*sin(ly)*(cos(cz)*sin(cy) 
                     - cos(cy)*sin(cx)*sin(cz)) - sin(lx)*(sin(cy)*sin(cz) + cos(cy)*cos(cz)*sin(cx));
        oy = atan2(srycrx / cos(ox), crycrx / cos(ox));

        //R_w_cur矩阵的R10（第2行第1列对应的元素)
        float srzcrx = sin(cx)*(cos(lz)*sin(ly) - cos(ly)*sin(lx)*sin(lz)) + cos(cx)*sin(cz)*(cos(ly)*cos(lz) 
                     + sin(lx)*sin(ly)*sin(lz)) + cos(lx)*cos(cx)*cos(cz)*sin(lz);
        //R_w_cur矩阵的R11（第2行第2列对应的元素)
        float crzcrx = cos(lx)*cos(lz)*cos(cx)*cos(cz) - cos(cx)*sin(cz)*(cos(ly)*sin(lz) 
                     - cos(lz)*sin(lx)*sin(ly)) - sin(cx)*(sin(ly)*sin(lz) + cos(ly)*cos(lz)*sin(lx));
        oz = atan2(srzcrx / cos(ox), crzcrx / cos(ox));
    }

    double rad2deg(double radians)
    {
        return radians * 180.0 / M_PI;
    }

    double deg2rad(double degrees)
    {
        return degrees * M_PI / 180.0;
    }

    void findCorrespondingCornerFeatures(int iterCount){

        int cornerPointsSharpNum = cornerPointsSharp->points.size();  //取出极大边线点的数量

        for (int i = 0; i < cornerPointsSharpNum; i++) {

            TransformToStart(&cornerPointsSharp->points[i], &pointSel);
            //每迭代五次，重新查找最近点
            if (iterCount % 5 == 0) {

                kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
                int closestPointInd = -1, minPointInd2 = -1;
                
                if (pointSearchSqDis[0] < nearestFeatureSearchSqDist) {   //const float nearestFeatureSearchSqDist = 25;
                    closestPointInd = pointSearchInd[0];
                    int closestPointScan = int(laserCloudCornerLast->points[closestPointInd].intensity);

                    float pointSqDis, minPointSqDis2 = nearestFeatureSearchSqDist;
                    for (int j = closestPointInd + 1; j < cornerPointsSharpNum; j++) {          //索引之后找
                        if (int(laserCloudCornerLast->points[j].intensity) > closestPointScan + 2.5) {        //不超过线束之上的两条线束
                            break; 
                        }

                        pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) * 
                                     (laserCloudCornerLast->points[j].x - pointSel.x) + 
                                     (laserCloudCornerLast->points[j].y - pointSel.y) * 
                                     (laserCloudCornerLast->points[j].y - pointSel.y) + 
                                     (laserCloudCornerLast->points[j].z - pointSel.z) * 
                                     (laserCloudCornerLast->points[j].z - pointSel.z);

                        if (int(laserCloudCornerLast->points[j].intensity) > closestPointScan) {
                            if (pointSqDis < minPointSqDis2) {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        }
                    }
                    for (int j = closestPointInd - 1; j >= 0; j--) {                                   //索引之前找
                        if (int(laserCloudCornerLast->points[j].intensity) < closestPointScan - 2.5) {  //不低于线束之下的两条线束
                            break;
                        }

                        pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) * 
                                     (laserCloudCornerLast->points[j].x - pointSel.x) + 
                                     (laserCloudCornerLast->points[j].y - pointSel.y) * 
                                     (laserCloudCornerLast->points[j].y - pointSel.y) + 
                                     (laserCloudCornerLast->points[j].z - pointSel.z) * 
                                     (laserCloudCornerLast->points[j].z - pointSel.z);

                        if (int(laserCloudCornerLast->points[j].intensity) < closestPointScan) {
                            if (pointSqDis < minPointSqDis2) {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        }
                    }
                }

                pointSearchCornerInd1[i] = closestPointInd;            //取出两个点的索引
                pointSearchCornerInd2[i] = minPointInd2;
            }

            if (pointSearchCornerInd2[i] >= 0) {

                tripod1 = laserCloudCornerLast->points[pointSearchCornerInd1[i]];
                tripod2 = laserCloudCornerLast->points[pointSearchCornerInd2[i]];

                float x0 = pointSel.x;
                float y0 = pointSel.y;
                float z0 = pointSel.z;
                float x1 = tripod1.x;
                float y1 = tripod1.y;
                float z1 = tripod1.z;
                float x2 = tripod2.x;
                float y2 = tripod2.y;
                float z2 = tripod2.z;

                float m11 = ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1));
                float m22 = ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1));   //?感觉少了个负号
                float m33 = ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1));

                float a012 = sqrt(m11 * m11  + m22 * m22 + m33 * m33);  //差乘向量的模长

                float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));  

                //*点线距离对点P(x0,y0,z0)的偏导
                float la =  ((y1 - y2)*m11 + (z1 - z2)*m22) / a012 / l12;     //对x0的偏导

                float lb = -((x1 - x2)*m11 - (z1 - z2)*m33) / a012 / l12;    //对y0的偏导
  
                float lc = -((x1 - x2)*m22 + (y1 - y2)*m33) / a012 / l12;    //对z0的偏导
                  
                float ld2 = a012 / l12;  //点到直线的距离

                float s = 1;
                if (iterCount >= 5) {
                    s = 1 - 1.8 * fabs(ld2); //影响因子(note:s越大，距离越近)
                }

                if (s > 0.1 && ld2 != 0) {
                    coeff.x = s * la;   //偏导的分量*s
                    coeff.y = s * lb;
                    coeff.z = s * lc;
                    coeff.intensity = s * ld2;  //点到直线的距离*s
                  
                    laserCloudOri->push_back(cornerPointsSharp->points[i]);
                    coeffSel->push_back(coeff);
                }
            }
        }
    }

    //todo 找对应的平面特征
    void findCorrespondingSurfFeatures(int iterCount){  

        int surfPointsFlatNum = surfPointsFlat->points.size();  //极小平面点的数量

        for (int i = 0; i < surfPointsFlatNum; i++) {    
            // 坐标变换到开始时刻，参数0是输入，参数1是输出
            TransformToStart(&surfPointsFlat->points[i], &pointSel);
            //每迭代五次，重新查找最近点
            if (iterCount % 5 == 0) {

                kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
                int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
                
                //平方距离小于25m
                if (pointSearchSqDis[0] < nearestFeatureSearchSqDist) {                  // const float nearestFeatureSearchSqDist = 25;
                    closestPointInd = pointSearchInd[0];
                    int closestPointScan = int(laserCloudSurfLast->points[closestPointInd].intensity);  //上一帧中kdtree最近邻点的线束号

                    float pointSqDis, minPointSqDis2 = nearestFeatureSearchSqDist, minPointSqDis3 = nearestFeatureSearchSqDist;
                    for (int j = closestPointInd + 1; j < surfPointsFlatNum; j++) {    //索引之后，线束之上找（不超过2条线束)
                        if (int(laserCloudSurfLast->points[j].intensity) > closestPointScan + 2.5) {
                            break;
                        }

                        pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) * 
                                     (laserCloudSurfLast->points[j].x - pointSel.x) + 
                                     (laserCloudSurfLast->points[j].y - pointSel.y) * 
                                     (laserCloudSurfLast->points[j].y - pointSel.y) + 
                                     (laserCloudSurfLast->points[j].z - pointSel.z) * 
                                     (laserCloudSurfLast->points[j].z - pointSel.z);

                        if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScan) {  //如果小于等于最近邻点的线束号（第2个点在当前线束上找)
                            if (pointSqDis < minPointSqDis2) {
                              minPointSqDis2 = pointSqDis;
                              minPointInd2 = j;
                            }
                        } else {                                                                                                                               //第3个点在当前线束之上找
                            if (pointSqDis < minPointSqDis3) {
                                minPointSqDis3 = pointSqDis;
                                minPointInd3 = j;
                            }
                        }
                    }
                    for (int j = closestPointInd - 1; j >= 0; j--) {        //索引之前，线束之下找（不超过2条线束)
                        if (int(laserCloudSurfLast->points[j].intensity) < closestPointScan - 2.5) {
                            break;
                        }

                        pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) * 
                                     (laserCloudSurfLast->points[j].x - pointSel.x) + 
                                     (laserCloudSurfLast->points[j].y - pointSel.y) * 
                                     (laserCloudSurfLast->points[j].y - pointSel.y) + 
                                     (laserCloudSurfLast->points[j].z - pointSel.z) * 
                                     (laserCloudSurfLast->points[j].z - pointSel.z);

                        if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScan) {
                            if (pointSqDis < minPointSqDis2) {
                                minPointSqDis2 = pointSqDis;
                                minPointInd2 = j;
                            }
                        } else {
                            if (pointSqDis < minPointSqDis3) {
                                minPointSqDis3 = pointSqDis;
                                minPointInd3 = j;
                            }
                        }
                    }
                }

                pointSearchSurfInd1[i] = closestPointInd;
                pointSearchSurfInd2[i] = minPointInd2;
                pointSearchSurfInd3[i] = minPointInd3;
            }

            // 前后都能找到对应的最近点在给定范围之内
            // 那么就开始计算距离
            // [pa,pb,pc]是tripod1，tripod2，tripod3这3个点构成的一个平面的方向量，
            // ps是模长，它是三角形面积的2倍
            if (pointSearchSurfInd2[i] >= 0 && pointSearchSurfInd3[i] >= 0) {

                tripod1 = laserCloudSurfLast->points[pointSearchSurfInd1[i]];
                tripod2 = laserCloudSurfLast->points[pointSearchSurfInd2[i]];
                tripod3 = laserCloudSurfLast->points[pointSearchSurfInd3[i]];
                //求平面Ax+By+Cz+D = 0;
                float pa = (tripod2.y - tripod1.y) * (tripod3.z - tripod1.z) 
                         - (tripod3.y - tripod1.y) * (tripod2.z - tripod1.z);
                float pb = (tripod2.z - tripod1.z) * (tripod3.x - tripod1.x) 
                         - (tripod3.z - tripod1.z) * (tripod2.x - tripod1.x);
                float pc = (tripod2.x - tripod1.x) * (tripod3.y - tripod1.y) 
                         - (tripod3.x - tripod1.x) * (tripod2.y - tripod1.y);
                float pd = -(pa * tripod1.x + pb * tripod1.y + pc * tripod1.z);
                //法向量的模
                float ps = sqrt(pa * pa + pb * pb + pc * pc);

                //pa,pb,pc为法向量各方向上的单位向量
                pa /= ps;
                pb /= ps;
                pc /= ps;
                pd /= ps;
                //求点到平面的距离
                float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

                float s = 1;
                if (iterCount >= 5) {
                    //加上影响因子(距离越远，s越小)
                    s = 1 - 1.8 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                            + pointSel.y * pointSel.y + pointSel.z * pointSel.z));
                }

                if (s > 0.1 && pd2 != 0) {
                    // [x,y,z]是整个平面的单位法量*s
                    // intensity是平面外一点到该平面的距离*s
                    coeff.x = s * pa;
                    coeff.y = s * pb;
                    coeff.z = s * pc;
                    coeff.intensity = s * pd2;
                    // 极小平面点放入laserCloudOri队列，距离*s，法向量*s值放入coeffSel
                    laserCloudOri->push_back(surfPointsFlat->points[i]);
                    coeffSel->push_back(coeff);
                }
            }
        }
    }

    bool calculateTransformationSurf(int iterCount){

        int pointSelNum = laserCloudOri->points.size();    //取出成功匹配的极小平面点的数量

        cv::Mat matA(pointSelNum, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(3, pointSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(3, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(3, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(3, 1, CV_32F, cv::Scalar::all(0));

        float srx = sin(transformCur[0]);               //transformCur: pitch,yaw,roll,x,y,z
        float crx = cos(transformCur[0]);
        float sry = sin(transformCur[1]);
        float cry = cos(transformCur[1]);
        float srz = sin(transformCur[2]);
        float crz = cos(transformCur[2]);
        float tx = transformCur[3];
        float ty = transformCur[4];
        float tz = transformCur[5];

        float a1 = crx*sry*srz; float a2 = crx*crz*sry; float a3 = srx*sry; float a4 = tx*a1 - ty*a2 - tz*a3;
        float a5 = srx*srz; float a6 = crz*srx; float a7 = ty*a6 - tz*crx - tx*a5;
        float a8 = crx*cry*srz; float a9 = crx*cry*crz; float a10 = cry*srx; float a11 = tz*a10 + ty*a9 - tx*a8;

        float b1 = -crz*sry - cry*srx*srz; float b2 = cry*crz*srx - sry*srz;
        float b5 = cry*crz - srx*sry*srz; float b6 = cry*srz + crz*srx*sry;

        float c1 = -b6; float c2 = b5; float c3 = tx*b6 - ty*b5; float c4 = -crx*crz; float c5 = crx*srz; float c6 = ty*c5 + tx*-c4;
        float c7 = b2; float c8 = -b1; float c9 = tx*-b2 - ty*-b1;
        
        // 构建雅可比矩阵，求解(对原先的pitch,roll,以及z进行约束和优化)(note:在平面上，x，y,和yaw不可观)
        //!    本质：      雅可比矩阵 = 点面距离对点状态的偏导 * 点状态对估计量的偏导
        for (int i = 0; i < pointSelNum; i++) {

            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];
            
            //todo  R_last_cur = Ry(-yaw)*Rx(-pitch)*Rz(-roll) = Ry(-ry)*Rx(-rx)*Rz(-rz) (和点云取畸变TransformToStart()函数里思路一样)（ P_k = R_k_k+1 * (P_k+1 - tk) )
            float arx = (-a1*pointOri.x + a2*pointOri.y + a3*pointOri.z + a4) * coeff.x       //原先pitch角
                      + (a5*pointOri.x - a6*pointOri.y + crx*pointOri.z + a7) * coeff.y
                      + (a8*pointOri.x - a9*pointOri.y - a10*pointOri.z + a11) * coeff.z;

            float arz = (c1*pointOri.x + c2*pointOri.y + c3) * coeff.x             //原先的roll角
                      + (c4*pointOri.x - c5*pointOri.y + c6) * coeff.y
                      + (c7*pointOri.x + c8*pointOri.y + c9) * coeff.z;

            float aty = -b6 * coeff.x + c4 * coeff.y + b2 * coeff.z;        //原先的z轴

            float d2 = coeff.intensity;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = arz;
            matA.at<float>(i, 2) = aty;
            matB.at<float>(i, 0) = -0.05 * d2;
        }

        //*构造JTJ以及-JTe矩阵
        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        //利用高斯牛顿求解
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        // iterCount==0 说明是第一次迭代，需要初始化
        if (iterCount == 0) {
            cv::Mat matE(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(3, 3, CV_32F, cv::Scalar::all(0));

            // 对近似的Hessian矩阵求特征值和特征向量， （Hessian矩阵：多元函数的二阶偏导数构成的矩阵）
            //matE特征值,matV是特征向量
            //*退化方向只与原始的约束方向  A有关，与原始约束的位置 b 无关     (线性化系统的结论)
            //算这个的目的是要判断退化，即约束中较小的偏移会导致解所在的局部区域发生较大的变化
            cv::eigen(matAtA, matE, matV);  //  参数1：输入的矩阵(要求是转置矩阵)   参数2：特征值(结果为从大到小)     参数3：特征向量(按行排列)
            matV.copyTo(matV2);

            isDegenerate = false;
            //初次优化时，特征值门限设置为10，小于这个值认为是退化了
            //系统退化与否和系统是否存在解没有必然联系，即使系统出现退化，系统也是可能存在解的，
            //*因此需要将系统的解进行调整，一个策略就是将解进行投影，
            //对于退化方向，使用优化的状态估计值，对于非退化方向，依然使用方程的解。
            //*另一个策略就是直接抛弃解在退化方向的分量。
            //对于退化方向，我们不考虑，直接丢弃，只考虑非退化方向解的增量。
            float eignThre[3] = {10, 10, 10};   //特征值取值门槛
            for (int i = 2; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) { //特征值太小，则认为处在兼并环境中，发生了退化
                    for (int j = 0; j < 3; j++) {  //对应的特征向量置为0
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            //计算P矩阵
            matP = matV.inv() * matV2;     // 非线性求解（只使用非线性优化解在非退化方向的分量，不考虑退化方向的分量）
        }

        if (isDegenerate) {  //如果发生退化，只使用预测矩阵P计算
            cv::Mat matX2(3, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        //累加每次迭代的旋转平移量
        transformCur[0] += matX.at<float>(0, 0);   //rx(原先pitch)
        transformCur[2] += matX.at<float>(1, 0);  //rz(原先roll）
        transformCur[4] += matX.at<float>(2, 0);   //ty(原先的z轴)

        for(int i=0; i<6; i++){
            if(isnan(transformCur[i]))  //判断是否非数字
                transformCur[i]=0;
        }
        //计算旋转平移量，如果很小就停止迭代
        float deltaR = sqrt(
                            pow(rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(rad2deg(matX.at<float>(1, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(2, 0) * 100, 2));

        if (deltaR < 0.1 && deltaT < 0.1) { //迭代终止条件
            return false;
        }
        return true;
    }

    bool calculateTransformationCorner(int iterCount){

        int pointSelNum = laserCloudOri->points.size();  //取出成功匹配的极大边线点的数量

        cv::Mat matA(pointSelNum, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(3, pointSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(3, 3, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(3, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(3, 1, CV_32F, cv::Scalar::all(0));

        float srx = sin(transformCur[0]);  //pitch
        float crx = cos(transformCur[0]);
        float sry = sin(transformCur[1]);  //yaw
        float cry = cos(transformCur[1]);
        float srz = sin(transformCur[2]);   //roll
        float crz = cos(transformCur[2]);
        float tx = transformCur[3];
        float ty = transformCur[4];
        float tz = transformCur[5];

        float b1 = -crz*sry - cry*srx*srz; float b2 = cry*crz*srx - sry*srz; float b3 = crx*cry; float b4 = tx*-b1 + ty*-b2 + tz*b3;
        float b5 = cry*crz - srx*sry*srz; float b6 = cry*srz + crz*srx*sry; float b7 = crx*sry; float b8 = tz*b7 - ty*b6 - tx*b5;

        float c5 = crx*srz;
        // 构建雅可比矩阵，求解(对原先的yaw,x,以及y进行约束和优化)(note:在角点上，yaw，x,和y可观)
        for (int i = 0; i < pointSelNum; i++) {  //极小边线点

            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];
           //todo  R_last_cur = Ry(-yaw)*Rx(-pitch)*Rz(-roll) = Ry(-ry)*Rx(-rx)*Rz(-rz) (和点云取畸变TransformToStart()函数里思路一样)（ P_k = R_k_k+1 * (P_k+1 - tk) )
            float ary = (b1*pointOri.x + b2*pointOri.y - b3*pointOri.z + b4) * coeff.x
                      + (b5*pointOri.x + b6*pointOri.y - b7*pointOri.z + b8) * coeff.z;

            float atx = -b5 * coeff.x + c5 * coeff.y + b1 * coeff.z;

            float atz = b7 * coeff.x - srx * coeff.y - b3 * coeff.z;

            float d2 = coeff.intensity;

            matA.at<float>(i, 0) = ary;
            matA.at<float>(i, 1) = atx;
            matA.at<float>(i, 2) = atz;
            matB.at<float>(i, 0) = -0.05 * d2;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        // iterCount==0 说明是第一次迭代，需要初始化
        if (iterCount == 0) {
            cv::Mat matE(1, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(3, 3, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(3, 3, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);    //  参数1：输入的矩阵(要求是转置矩阵)   参数2：特征值(结果为从大到小)     参数3：特征向量(按行排列)
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[3] = {10, 10, 10};
            for (int i = 2; i >= 0; i--) {   //特征值从小到大遍历
                if (matE.at<float>(0, i) < eignThre[i]) {     //特征值小于阈值则将其对应的特征向量置为0
                    for (int j = 0; j < 3; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;  //构建P矩阵   // 非线性求解（只使用非线性优化解在非退化方向的分量，不考虑退化方向的分量）
        }

        if (isDegenerate) {  //如果发生退化，只使用预测矩阵P计算
            cv::Mat matX2(3, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;  
        }

        //累加每次迭代的旋转平移量
        transformCur[1] += matX.at<float>(0, 0);  //ry(原先的yaw)
        transformCur[3] += matX.at<float>(1, 0);  //tx(原先的y)
        transformCur[5] += matX.at<float>(2, 0);   //tz(原先的x)

        for(int i=0; i<6; i++){
            if(isnan(transformCur[i]))    //判断是否非数字
                transformCur[i]=0;
        }
        //计算旋转平移量，如果很小就停止迭代
        float deltaR = sqrt(
                            pow(rad2deg(matX.at<float>(0, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(1, 0) * 100, 2) +
                            pow(matX.at<float>(2, 0) * 100, 2));

        if (deltaR < 0.1 && deltaT < 0.1) { //迭代终止条件
            return false;
        }
        return true;
    }

    bool calculateTransformation(int iterCount){

        int pointSelNum = laserCloudOri->points.size();

        cv::Mat matA(pointSelNum, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matAt(6, pointSelNum, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
        cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

        float srx = sin(transformCur[0]);
        float crx = cos(transformCur[0]);
        float sry = sin(transformCur[1]);
        float cry = cos(transformCur[1]);
        float srz = sin(transformCur[2]);
        float crz = cos(transformCur[2]);
        float tx = transformCur[3];
        float ty = transformCur[4];
        float tz = transformCur[5];

        float a1 = crx*sry*srz; float a2 = crx*crz*sry; float a3 = srx*sry; float a4 = tx*a1 - ty*a2 - tz*a3;
        float a5 = srx*srz; float a6 = crz*srx; float a7 = ty*a6 - tz*crx - tx*a5;
        float a8 = crx*cry*srz; float a9 = crx*cry*crz; float a10 = cry*srx; float a11 = tz*a10 + ty*a9 - tx*a8;

        float b1 = -crz*sry - cry*srx*srz; float b2 = cry*crz*srx - sry*srz; float b3 = crx*cry; float b4 = tx*-b1 + ty*-b2 + tz*b3;
        float b5 = cry*crz - srx*sry*srz; float b6 = cry*srz + crz*srx*sry; float b7 = crx*sry; float b8 = tz*b7 - ty*b6 - tx*b5;

        float c1 = -b6; float c2 = b5; float c3 = tx*b6 - ty*b5; float c4 = -crx*crz; float c5 = crx*srz; float c6 = ty*c5 + tx*-c4;
        float c7 = b2; float c8 = -b1; float c9 = tx*-b2 - ty*-b1;

        for (int i = 0; i < pointSelNum; i++) {

            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            float arx = (-a1*pointOri.x + a2*pointOri.y + a3*pointOri.z + a4) * coeff.x
                      + (a5*pointOri.x - a6*pointOri.y + crx*pointOri.z + a7) * coeff.y
                      + (a8*pointOri.x - a9*pointOri.y - a10*pointOri.z + a11) * coeff.z;

            float ary = (b1*pointOri.x + b2*pointOri.y - b3*pointOri.z + b4) * coeff.x
                      + (b5*pointOri.x + b6*pointOri.y - b7*pointOri.z + b8) * coeff.z;

            float arz = (c1*pointOri.x + c2*pointOri.y + c3) * coeff.x
                      + (c4*pointOri.x - c5*pointOri.y + c6) * coeff.y
                      + (c7*pointOri.x + c8*pointOri.y + c9) * coeff.z;

            float atx = -b5 * coeff.x + c5 * coeff.y + b1 * coeff.z;

            float aty = -b6 * coeff.x + c4 * coeff.y + b2 * coeff.z;

            float atz = b7 * coeff.x - srx * coeff.y - b3 * coeff.z;

            float d2 = coeff.intensity;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;
            matA.at<float>(i, 3) = atx;
            matA.at<float>(i, 4) = aty;
            matA.at<float>(i, 5) = atz;
            matB.at<float>(i, 0) = -0.05 * d2;
        }

        cv::transpose(matA, matAt);
        matAtA = matAt * matA;
        matAtB = matAt * matB;
        cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

        if (iterCount == 0) {
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            float eignThre[6] = {10, 10, 10, 10, 10, 10};
            for (int i = 5; i >= 0; i--) {
                if (matE.at<float>(0, i) < eignThre[i]) {
                    for (int j = 0; j < 6; j++) {
                        matV2.at<float>(i, j) = 0;
                    }
                    isDegenerate = true;
                } else {
                    break;
                }
            }
            matP = matV.inv() * matV2;
        }

        if (isDegenerate) {
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
        }

        transformCur[0] += matX.at<float>(0, 0);
        transformCur[1] += matX.at<float>(1, 0);
        transformCur[2] += matX.at<float>(2, 0);
        transformCur[3] += matX.at<float>(3, 0);
        transformCur[4] += matX.at<float>(4, 0);
        transformCur[5] += matX.at<float>(5, 0);

        for(int i=0; i<6; i++){
            if(isnan(transformCur[i]))
                transformCur[i]=0;
        }

        float deltaR = sqrt(
                            pow(rad2deg(matX.at<float>(0, 0)), 2) +
                            pow(rad2deg(matX.at<float>(1, 0)), 2) +
                            pow(rad2deg(matX.at<float>(2, 0)), 2));
        float deltaT = sqrt(
                            pow(matX.at<float>(3, 0) * 100, 2) +
                            pow(matX.at<float>(4, 0) * 100, 2) +
                            pow(matX.at<float>(5, 0) * 100, 2));

        if (deltaR < 0.1 && deltaT < 0.1) {
            return false;
        }
        return true;
    }

    void checkSystemInitialization(){
        // 当前次极大边线点和次极小平面点就变成了上一帧的边线点和平面点(用于构建kdtree，从而方便构建点线和点面优化问题)，把索引和数量都转移过去
        pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
        cornerPointsLessSharp = laserCloudCornerLast;
        laserCloudCornerLast = laserCloudTemp;

        laserCloudTemp = surfPointsLessFlat;
        surfPointsLessFlat = laserCloudSurfLast;
        laserCloudSurfLast = laserCloudTemp;

        kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
        kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

        laserCloudCornerLastNum = laserCloudCornerLast->points.size();
        laserCloudSurfLastNum = laserCloudSurfLast->points.size();

        sensor_msgs::PointCloud2 laserCloudCornerLast2;
        pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
        laserCloudCornerLast2.header.stamp = cloudHeader.stamp;
        laserCloudCornerLast2.header.frame_id = "/camera";
        pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

        sensor_msgs::PointCloud2 laserCloudSurfLast2;
        pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
        laserCloudSurfLast2.header.stamp = cloudHeader.stamp;
        laserCloudSurfLast2.header.frame_id = "/camera";
        pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

        //todo 第一帧数据的初始点姿态角作为全局坐标系姿态角
        transformSum[0] += imuPitchStart; // transformSum数组初始值为0，imuPitchStart为每帧imu起始的pitch角度大小 (note:transformSum数组内容为：pitch,yaw,roll,x,y,z)
        transformSum[2] += imuRollStart;   //imuRollStart为每帧imu起始的pitch角度大小 (世界坐标系)

        systemInitedLM = true;
    }

    void updateInitialGuess(){  //将当前时刻保存的imu数据作为先验数据
        //todo 这里主要是保存当前点云中最后一个点的旋转角、最后一个点相对于第一个点的位移以及速度
        imuPitchLast = imuPitchCur;  //世界坐标系下的欧拉角
        imuYawLast = imuYawCur;
        imuRollLast = imuRollCur;

        imuShiftFromStartX = imuShiftFromStartXCur;  //相对于每帧初始时刻i=0时的相对位移（imu起始坐标系，即StartIMU)
        imuShiftFromStartY = imuShiftFromStartYCur;
        imuShiftFromStartZ = imuShiftFromStartZCur;

        imuVeloFromStartX = imuVeloFromStartXCur;  //相对于每帧初始时刻i=0时的相对速度（imu起始坐标系,即StartIMU)
        imuVeloFromStartY = imuVeloFromStartYCur;  
        imuVeloFromStartZ = imuVeloFromStartZCur;
        
        //更新旋转量（相邻两帧）：
            // 关于下面负号的说明：
            // transformCur是在Cur坐标系下的 p_start=R*p_cur+t
            // R和t是在Cur坐标系下的
            // 而imuAngularFromStart是在start坐标系下的，所以需要加负号

        //* 而imuAngularFromStart是在世界坐标下的值
        if (imuAngularFromStartX != 0 || imuAngularFromStartY != 0 || imuAngularFromStartZ != 0){  // 距离上一次插补，旋转过的角度变化值（即:两帧之间的角度变化量)
            transformCur[0] = - imuAngularFromStartY;              //转到后一帧中
            transformCur[1] = - imuAngularFromStartZ;
            transformCur[2] = - imuAngularFromStartX;
        }
        
        //更新位移量（两帧之间）：
             // 速度乘以时间，当前变换中的位移
        if (imuVeloFromStartX != 0 || imuVeloFromStartY != 0 || imuVeloFromStartZ != 0){
            transformCur[3] -= imuVeloFromStartX * scanPeriod;
            transformCur[4] -= imuVeloFromStartY * scanPeriod;
            transformCur[5] -= imuVeloFromStartZ * scanPeriod;
        }
    }

    
    void updateTransformation(){     

        if (laserCloudCornerLastNum < 10 || laserCloudSurfLastNum < 100)  //如果角点和面点数量太少，就返回
            return;

        //Levenberg-Marquardt算法(L-M method)，非线性最小二乘算法，最优化算法的一种
        //最多迭代25次
        for (int iterCount1 = 0; iterCount1 < 25; iterCount1++) {
            laserCloudOri->clear();
            coeffSel->clear();

            // 找到对应的特征平面
            // 然后计算协方差矩阵，保存在coeffSel队列中
            // laserCloudOri中保存的是对应于coeffSel的未转换到开始时刻的原始点云数据
            findCorrespondingSurfFeatures(iterCount1);

            if (laserCloudOri->points.size() < 10)  //特征匹配的数量太少(极小平面点)
                continue;
            // 通过面特征的匹配，计算变换矩阵
            if (calculateTransformationSurf(iterCount1) == false)
                break;
        }

        for (int iterCount2 = 0; iterCount2 < 25; iterCount2++) {  

            laserCloudOri->clear();
            coeffSel->clear();

            // 找到对应的特征边/角点
            // 寻找边特征的方法和寻找平面特征的很类似，过程可以参照寻找平面特征的注释
            findCorrespondingCornerFeatures(iterCount2);

            if (laserCloudOri->points.size() < 10)  //特征匹配的数量太小(极大边线点)
                continue;
            // 通过角/边特征的匹配，计算变换矩阵
            if (calculateTransformationCorner(iterCount2) == false)
                break;
        }
    }

    // 积分总变换                  参考文章链接：https://blog.csdn.net/l1323/article/details/106335035               https://zhuanlan.zhihu.com/p/589732149
    void integrateTransformation(){
        float rx, ry, rz, tx, ty, tz;   //*当前帧起始位置的lidar全局位姿
        // AccumulateRotation作用
        //todo 将计算的两帧之间的位姿“累加”起来，获得相对于第一帧的旋转矩阵(欧拉角rx,ry,rz为世界坐标系下的角度值)
        // transformSum + (-transformCur) =(rx,ry,rz)
        AccumulateRotation(transformSum[0], transformSum[1], transformSum[2], 
                           -transformCur[0], -transformCur[1], -transformCur[2], rx, ry, rz);

        // 进行平移分量的更新  
        //R^T  = R_transformCur = Ry(-yaw)*Rx(-pitch)*Rz(-roll)         //transformCur是以当前点为坐标系
        //                                                                            [  R_transforSum           t_sum     ]           [     R^T                         -R^T*t    ]        [  R_transforSum*R^T         t_sum- R_transforSum*R^T    ]
        //T = T_transforsum * T_transforCur  =                                                                      *                                                                 =  
        //                                                                            [          0                                 1               ]           [        0                                1          ]        [                  0                                                            1                              ]

        float x1 = cos(rz) * (transformCur[3] - imuShiftFromStartX) 
                 - sin(rz) * (transformCur[4] - imuShiftFromStartY);
        float y1 = sin(rz) * (transformCur[3] - imuShiftFromStartX) 
                 + cos(rz) * (transformCur[4] - imuShiftFromStartY);
        float z1 = transformCur[5] - imuShiftFromStartZ;

        float x2 = x1;
        float y2 = cos(rx) * y1 - sin(rx) * z1;
        float z2 = sin(rx) * y1 + cos(rx) * z1;

        //求相对于原点的平移量
        tx = transformSum[3] - (cos(ry) * x2 + sin(ry) * z2);
        ty = transformSum[4] - y2;
        tz = transformSum[5] - (-sin(ry) * x2 + cos(ry) * z2);

        //*根据IMU修正旋转量(考虑惯导当前帧首末时刻姿态差得到最终lidar全局姿态)  （前面计算了k+1时刻雷达帧起点得姿态，然后该函数，通过IMU数据计算该帧点云最后一个点得姿态）
        PluginIMURotation(rx, ry, rz, imuPitchStart, imuYawStart, imuRollStart,            
                          imuPitchLast, imuYawLast, imuRollLast, rx, ry, rz);

        //得到世界坐标系下的转移矩阵
        transformSum[0] = rx;
        transformSum[1] = ry;
        transformSum[2] = rz;
        transformSum[3] = tx;
        transformSum[4] = ty;
        transformSum[5] = tz;
    }

    void publishOdometry(){  //todo 发布里程计位姿                       transformSum  ：pitch  yaw roll tx ty tz
        geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(transformSum[2], -transformSum[0], -transformSum[1]);
        // rx,ry,rz转化为四元数发布  (世界坐标系--->右手坐标系)
        laserOdometry.header.stamp = cloudHeader.stamp;
        laserOdometry.pose.pose.orientation.x = -geoQuat.y;
        laserOdometry.pose.pose.orientation.y = -geoQuat.z;
        laserOdometry.pose.pose.orientation.z = geoQuat.x;
        laserOdometry.pose.pose.orientation.w = geoQuat.w;
        laserOdometry.pose.pose.position.x = transformSum[3];
        laserOdometry.pose.pose.position.y = transformSum[4];
        laserOdometry.pose.pose.position.z = transformSum[5];
        pubLaserOdometry.publish(laserOdometry);
        // laserOdometryTrans 是用于tf广播         tf广播参考链接：http://wiki.ros.org/tf/Tutorials/Writing%20a%20tf%20broadcaster%20%28C%2B%2B%29
        laserOdometryTrans.stamp_ = cloudHeader.stamp;
        laserOdometryTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
        laserOdometryTrans.setOrigin(tf::Vector3(transformSum[3], transformSum[4], transformSum[5]));
        tfBroadcaster.sendTransform(laserOdometryTrans);
    }

    void adjustOutlierCloud(){  //异常点云进行坐标调整（激光坐标系-->右手坐标系)
        PointType point;
        int cloudSize = outlierCloud->points.size();
        for (int i = 0; i < cloudSize; ++i)
        {
            point.x = outlierCloud->points[i].y;
            point.y = outlierCloud->points[i].z;
            point.z = outlierCloud->points[i].x;
            point.intensity = outlierCloud->points[i].intensity;
            outlierCloud->points[i] = point;
        }
    }

    void publishCloudsLast(){

        updateImuRollPitchYawStartSinCos();   // 更新初始时刻i=0时刻的rpy角的正余弦值

        int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
        for (int i = 0; i < cornerPointsLessSharpNum; i++) {
            // TransformToEnd的作用是将k+1时刻的less特征点转移至k+1时刻的sweep的结束位置处的雷达坐标系下
            TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);  //note:点刚开始在当前帧的startIMU坐标系下--->imuEnd坐标系下
        }


        int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
        for (int i = 0; i < surfPointsLessFlatNum; i++) {
            TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
        }

        pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
        cornerPointsLessSharp = laserCloudCornerLast;
        laserCloudCornerLast = laserCloudTemp;

        laserCloudTemp = surfPointsLessFlat;
        surfPointsLessFlat = laserCloudSurfLast;
        laserCloudSurfLast = laserCloudTemp;

        laserCloudCornerLastNum = laserCloudCornerLast->points.size();
        laserCloudSurfLastNum = laserCloudSurfLast->points.size();

        if (laserCloudCornerLastNum > 10 && laserCloudSurfLastNum > 100) {  //次极大边线点大于10并且次极小平面点大于100，构建kdtree
            kdtreeCornerLast->setInputCloud(laserCloudCornerLast);  //将变换后的点云作为下一帧点云数据的参考帧
            kdtreeSurfLast->setInputCloud(laserCloudSurfLast);
        }

        frameCount++;     //frameCount初始值为1

        if (frameCount >= skipFrameNum + 1) {        //skipFrameNum初始值为1
 
            frameCount = 0;

            adjustOutlierCloud();
            //发布异常点云
            sensor_msgs::PointCloud2 outlierCloudLast2;
            pcl::toROSMsg(*outlierCloud, outlierCloudLast2);
            outlierCloudLast2.header.stamp = cloudHeader.stamp;
            outlierCloudLast2.header.frame_id = "/camera";
            pubOutlierCloudLast.publish(outlierCloudLast2);  

            //发布当前帧的极小边线点
            sensor_msgs::PointCloud2 laserCloudCornerLast2;
            pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
            laserCloudCornerLast2.header.stamp = cloudHeader.stamp;
            laserCloudCornerLast2.header.frame_id = "/camera";
            pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

            //发布当前帧的极小平面点
            sensor_msgs::PointCloud2 laserCloudSurfLast2;
            pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
            laserCloudSurfLast2.header.stamp = cloudHeader.stamp;
            laserCloudSurfLast2.header.frame_id = "/camera";
            pubLaserCloudSurfLast.publish(laserCloudSurfLast2);
        }
    }

    //! 核心函数
    void runFeatureAssociation()
    {
        //如果有新数据进来则执行，否则不执行任何操作
        if (newSegmentedCloud && newSegmentedCloudInfo && newOutlierCloud &&      //回调函数中，将其设置为了true
            std::abs(timeNewSegmentedCloudInfo - timeNewSegmentedCloud) < 0.05 &&
            std::abs(timeNewOutlierCloud - timeNewSegmentedCloud) < 0.05){

            newSegmentedCloud = false;
            newSegmentedCloudInfo = false;
            newOutlierCloud = false;
        }else{
            return;
        }
        /**
        	1. Feature Extraction
        */
        adjustDistortion();  // 主要进行的处理是将点云数据进行坐标变换，进行插补等工作（差补到imu的起始坐标系)(去除了非匀速运动造成的误差)

        calculateSmoothness();  // 不完全按照公式进行光滑性计算，并保存结果

        // 标记阻塞点??? 阻塞点是什么点???
        // 参考了csdn若愚maimai大佬的博客，这里的阻塞点指过近的点
        // 指在点云中可能出现的互相遮挡的情况
        markOccludedPoints();   //标记遮挡的点和平行激光束的点

        // 特征抽取，然后分别保存到cornerPointsSharp等等队列中去
        // 保存到不同的队列是不同类型的点云，进行了标记的工作，
        // 这一步中减少了点云数量，使计算量减少
        extractFeatures();

        // 发布cornerPointsSharp等4种类型的点云数据
        publishCloud(); // cloud for visualization
	
        /**
		2. Feature Association
        */
        if (!systemInitedLM) {   //systemInitedLM初始值为false
            checkSystemInitialization();
            return;
        }

        // 预测位姿
        updateInitialGuess();

        // 更新变换
        updateTransformation();

        // 积分总变换
        integrateTransformation();

        publishOdometry();  //发布里程计位姿

        publishCloudsLast(); // cloud to mapOptimization（发布异常点云+用于构建kdtree的角点(当前帧的次极大边线点)和面点(当前帧的次极小平面点))
    }
};




int main(int argc, char** argv)
{
    ros::init(argc, argv, "lego_loam");

    ROS_INFO("\033[1;32m---->\033[0m Feature Association Started.");

    FeatureAssociation FA;

    ros::Rate rate(200);
    while (ros::ok())
    {
        ros::spinOnce();

        FA.runFeatureAssociation();

        rate.sleep();
    }
    
    ros::spin();
    return 0;
}
