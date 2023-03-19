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

class ImageProjection{
private:

    ros::NodeHandle nh;

    ros::Subscriber subLaserCloud;
    
    ros::Publisher pubFullCloud;
    ros::Publisher pubFullInfoCloud;

    ros::Publisher pubGroundCloud;
    ros::Publisher pubSegmentedCloud;
    ros::Publisher pubSegmentedCloudPure;
    ros::Publisher pubSegmentedCloudInfo;
    ros::Publisher pubOutlierCloud;

    pcl::PointCloud<PointType>::Ptr laserCloudIn;
    pcl::PointCloud<PointXYZIR>::Ptr laserCloudInRing;

    pcl::PointCloud<PointType>::Ptr fullCloud; // projected velodyne raw cloud, but saved in the form of 1-D matrix
    pcl::PointCloud<PointType>::Ptr fullInfoCloud; // same as fullCloud, but with intensity - range

    pcl::PointCloud<PointType>::Ptr groundCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloudPure;
    pcl::PointCloud<PointType>::Ptr outlierCloud;

    PointType nanPoint; // fill in fullCloud at each iteration

    cv::Mat rangeMat; // range matrix for range image
    cv::Mat labelMat; // label matrix for segmentaiton marking
    cv::Mat groundMat; // ground matrix for ground cloud marking
    int labelCount;

    float startOrientation;
    float endOrientation;

    cloud_msgs::cloud_info segMsg; // info of segmented cloud
    std_msgs::Header cloudHeader;

    std::vector<std::pair<int8_t, int8_t> > neighborIterator; // neighbor iterator for segmentaiton process

    uint16_t *allPushedIndX; // array for tracking points of a segmented object
    uint16_t *allPushedIndY;

    uint16_t *queueIndX; // array for breadth-first search process of segmentation, for speed
    uint16_t *queueIndY;

public:
    ImageProjection(): //构造函数初始化
        nh("~"){
        // 订阅来自激光雷达驱动的topic 
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 1, &ImageProjection::cloudHandler, this);

        pubFullCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_projected", 1);
        pubFullInfoCloud = nh.advertise<sensor_msgs::PointCloud2> ("/full_cloud_info", 1);

        pubGroundCloud = nh.advertise<sensor_msgs::PointCloud2> ("/ground_cloud", 1);
        pubSegmentedCloud = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud", 1);
        pubSegmentedCloudPure = nh.advertise<sensor_msgs::PointCloud2> ("/segmented_cloud_pure", 1);
        pubSegmentedCloudInfo = nh.advertise<cloud_msgs::cloud_info> ("/segmented_cloud_info", 1);   //自定义点云信息
        pubOutlierCloud = nh.advertise<sensor_msgs::PointCloud2> ("/outlier_cloud", 1);

        nanPoint.x = std::numeric_limits<float>::quiet_NaN();   //生成nan值，即nanPoint.x = nan;
        nanPoint.y = std::numeric_limits<float>::quiet_NaN();
        nanPoint.z = std::numeric_limits<float>::quiet_NaN();
        nanPoint.intensity = -1;

        allocateMemory();    // 初始化各类参数以及分配内存
        resetParameters();  // 初始化/重置各类参数内容
    }

    // 初始化各类参数以及分配内存
    void allocateMemory(){
        //指向开辟的堆区，laserCloudIn(new pcl::PointCloud<PointType>()) = laserCloudIn.reset(new pcl::PointCloud<PointType>())
        laserCloudIn.reset(new pcl::PointCloud<PointType>());
        laserCloudInRing.reset(new pcl::PointCloud<PointXYZIR>());

        fullCloud.reset(new pcl::PointCloud<PointType>());
        fullInfoCloud.reset(new pcl::PointCloud<PointType>());

        groundCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloud.reset(new pcl::PointCloud<PointType>());
        segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
        outlierCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);   //重新指定容器的大小（16*1800),并用默认值0填充    
        fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

        segMsg.startRingIndex.assign(N_SCAN, 0);     //将16个0赋值给本身，即数组初始化为0
        segMsg.endRingIndex.assign(N_SCAN, 0);  

        segMsg.segmentedCloudGroundFlag.assign(N_SCAN*Horizon_SCAN, false); //同理
        segMsg.segmentedCloudColInd.assign(N_SCAN*Horizon_SCAN, 0);
        segMsg.segmentedCloudRange.assign(N_SCAN*Horizon_SCAN, 0);

        // labelComponents函数中用到了这个矩阵
		// 该矩阵用于求某个点的上下左右4个邻接点
        std::pair<int8_t, int8_t> neighbor;      //对组
        neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor); //点的上边      //行不同，列相同
        neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor);  //点的右边      //行相同，列不同
        neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor);  //点的左边
        neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor);  //点的下边

        allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];  //指针开辟堆区
        allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

        queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
        queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];
    }
 
    // 初始化/重置各类参数内容
    void resetParameters(){
        laserCloudIn->clear();  //指针清空
        groundCloud->clear();
        segmentedCloud->clear();
        segmentedCloudPure->clear();
        outlierCloud->clear();

        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));   //32位浮点数，单通道，图模型矩阵全用最大值填充
        groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));     //8位有符号整型，地面矩阵全用0进行填充
        labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));    //32位有符号整形，标签矩阵全用0进行填充
        labelCount = 1;

        std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);    //用nan点填充fullCloud点云
        std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint); //同理
    }

    ~ImageProjection(){}   //析构函数

    void copyPointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){
        cloudHeader = laserCloudMsg->header;  //取出时间戳
        // cloudHeader.stamp = ros::Time::now(); // Ouster lidar users may need to uncomment this line
        pcl::fromROSMsg(*laserCloudMsg, *laserCloudIn); // 将ROS中的sensor_msgs::PointCloud2ConstPtr类型转换到pcl点云库指针(laserCloudIn里存放的是当前帧点云)
        // Remove Nan points
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);     //cloud_out.points[i] = cloud_in.points[index[i]]
        // have "ring" channel in the cloud（线束通道可用时)
        if (useCloudRing == true){
            pcl::fromROSMsg(*laserCloudMsg, *laserCloudInRing);   //将ros消息点云格式转成pcl点云格式
            if (laserCloudInRing->is_dense == false) {  //如果点云中的数据不是有限的，打印错误，中断程序
                ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
                ros::shutdown();
            }  
        }
    }
    
    //! 订阅点云的回调函数(主要内容)
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg){

        // 1. Convert ros message to pcl point cloud
        copyPointCloud(laserCloudMsg);   
        // 2. Start and end angle of a scan
        findStartEndAngle();  //计算当前帧起始角，结束角，以及两者之间的角度差
        // 3. Range image projection
        projectPointCloud();    //算出点的索引，然后将点距离lidar中心的距离，存放在对应索引的图模型矩阵中
        // 4. Mark ground points
        groundRemoval();    //根据相邻线束之间的夹角是否小于10度，标记地面点
        // 5. Point cloud segmentation
        cloudSegmentation();  //点云聚类
        // 6. Publish all clouds
        publishCloud();    //(分割信息的点云+异常点+带地面的分割点云+全部点云+地面点+不带地面的分割点云+全部信息点云)
        // 7. Reset parameters for next iteration
        resetParameters();   
    }

    void findStartEndAngle(){
        //*start and end orientation of this cloud
        // 雷达坐标系：右->X,前->Y,上->Z
        // 雷达内部旋转扫描方向：Z轴俯视下来，顺时针方向（Z轴右手定则反向）

        // atan2(y,x)函数的返回值范围(-PI,PI],表示与复数x+yi的幅角
        // segMsg.startOrientation范围为(-PI,PI]
        // segMsg.endOrientation范围为(PI,3PI]
        // 因为内部雷达旋转方向原因，所以atan2(..)函数前面需要加一个负号（角度逆时针为正值)
        segMsg.startOrientation = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);    //起始角
        segMsg.endOrientation   = -atan2(laserCloudIn->points[laserCloudIn->points.size() - 1].y,
                                                     laserCloudIn->points[laserCloudIn->points.size() - 1].x) + 2 * M_PI;       //结束角
        // 开始和结束的角度差一般是多少？
		// 一个velodyne 雷达数据包转过的角度多大？
        // 雷达一般包含的是一圈的数据，所以角度差一般是2*PI，一个数据包转过360度

		// segMsg.endOrientation - segMsg.startOrientation范围为(0,4PI)
        // 如果角度差大于3Pi或小于Pi，说明角度差有问题，进行调整。
        if (segMsg.endOrientation - segMsg.startOrientation > 3 * M_PI) {
            segMsg.endOrientation -= 2 * M_PI;
        } else if (segMsg.endOrientation - segMsg.startOrientation < M_PI)
            segMsg.endOrientation += 2 * M_PI;
        // segMsg.orientationDiff的范围为(PI,3PI),一圈大小为2PI，应该在2PI左右
        segMsg.orientationDiff = segMsg.endOrientation - segMsg.startOrientation;  //计算结束与起始的角度差
    }

    void projectPointCloud(){  //距离图像投影
        // range image projection
        float verticalAngle, horizonAngle, range;
        size_t rowIdn, columnIdn, index, cloudSize; 
        PointType thisPoint;

        cloudSize = laserCloudIn->points.size();  //当前帧的点数

        for (size_t i = 0; i < cloudSize; ++i){  //遍历当前帧的每一个点
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            // find the row and column index in the image for this point  (图数据结构中，行索引范围：0～16， 列索引范围：0～1800)
            if (useCloudRing == true){    //如果线束信息可用，直接算出行索引 
                rowIdn = laserCloudInRing->points[i].ring;
            }
            else{
                verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;  // 计算竖直方向上的角度（雷达的第几线）
                rowIdn = (verticalAngle + ang_bottom) / ang_res_y;           //对16线来说（垂直角度-(-15))/2  = (垂直角度+15)/2
            }
            if (rowIdn < 0 || rowIdn >= N_SCAN)     //跳过不符合线号的点
                continue;

            // atan2(y,x)函数的返回值范围(-PI,PI],表示与复数x+yi的幅角
            // 下方角度atan2(..)交换了x和y的位置，计算的是与y轴正方向的夹角大小(关于y=x做对称变换)
            //* 这里是在雷达坐标系，所以是与正前方的夹角大小
            horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;      //计算水平角度,horizonAngle 为[-180,180]

           //计算水平线束id,转换到x负方向e为起始，顺时针为正方向，范围[0~H]
            columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;  //即把horizonAngle从[-180,180]映射到[450,2250]
                   /*角度变化
                       (-180)   -----   (-90)  ------  0  ------ 90 -------180
                       变为:  90(-270) ----180(-180) ---- (-90)  ----- (0)    ----- 90
                    */     
            //                   3/4*H
            //                   | y+ (0度)
            //                          |
            // (-90) (x-)H---------->H/2 (x+) （90度）
            //                          |
            //                          | y-
            //              5/4*H   H/4 （180度)
               
            if (columnIdn >= Horizon_SCAN)  //大于等于1800，则减去1800，相当于把1800～2250映射到0～450 (将它的范围转换到了[0,1800) )
                columnIdn -= Horizon_SCAN;
            // 经过上面columnIdn -= Horizon_SCAN的变换后的columnIdn分布：
            //          3/4*H
            //          | y+ 
            //     H    |
            // (x-)---------->H/2 (x+)
            //     0    |
            //          | y-
            //         H/4
            //对水平id进行检查
            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)   //范围必须为[0,1800）
                continue;

            range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);  //range:点距离lidar中心的距离
            if (range < sensorMinimumRange)   //距离小于1米就跳过
                continue;
            
            rangeMat.at<float>(rowIdn, columnIdn) = range;   //在图模型矩阵中，根据点的行和列索引填充点到lidar中心的距离

            thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;   //点的强度值=行号(整数)+列索引(小数部分)

            index = columnIdn  + rowIdn * Horizon_SCAN;  //算出该点的索引
            fullCloud->points[index] = thisPoint;                //在fullCloud指向的点云中，根据点的索引，存放点
            fullInfoCloud->points[index] = thisPoint;
            fullInfoCloud->points[index].intensity = range; // the corresponding range of a point is saved as "intensity"  //强度为点距离雷达中心的距离
        }
    }


    void groundRemoval(){
        size_t lowerInd, upperInd;
        float diffX, diffY, diffZ, angle;
        // groundMat
        // -1, no valid info to check if ground of not
        //  0, initial value, after validation, means not ground
        //  1, ground
        for (size_t j = 0; j < Horizon_SCAN; ++j){
            for (size_t i = 0; i < groundScanInd; ++i){          // groundScanInd 是在 utility.h 文件中声明的线数，groundScanInd=7

                lowerInd = j + ( i )*Horizon_SCAN;
                upperInd = j + (i+1)*Horizon_SCAN;

                // 初始化的时候用nanPoint.intensity = -1 填充
                // 都是-1 证明是空点nanPoint
                if (fullCloud->points[lowerInd].intensity == -1 ||
                    fullCloud->points[upperInd].intensity == -1){                //图模型矩阵中点的强度值 = 行号(整数)+列索引(小数部分)
                    // no info to check, invalid points
                    groundMat.at<int8_t>(i,j) = -1;            //地面矩阵
                    continue;
                }
                    
                // 由上下两线之间点的XYZ位置得到两线之间的俯仰角
				// 如果俯仰角在10度以内，则判定(i,j)为地面点,groundMat[i][j]=1
				// 否则，则不是地面点，进行后续操作
                diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
                diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
                diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

                angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;

                if (abs(angle - sensorMountAngle) <= 10){       //extern const float sensorMountAngle = 0.0;如果相邻两线之间的夹角<10度，则认为是地面点
                    groundMat.at<int8_t>(i,j) = 1;
                    groundMat.at<int8_t>(i+1,j) = 1;
                }
            }
        }
        // extract ground cloud (groundMat == 1)
        // mark entry that doesn't need to label (ground and invalid point) for segmentation
        // note that ground remove is from 0~N_SCAN-1, need rangeMat for mark label matrix for the 16th scan

        // 找到所有点中的地面点或者距离为FLT_MAX(rangeMat的初始值)的点，并将他们标记为-1
		// rangeMat[i][j]==FLT_MAX，代表的含义是什么？ 无效点
        for (size_t i = 0; i < N_SCAN; ++i){
            for (size_t j = 0; j < Horizon_SCAN; ++j){
                if (groundMat.at<int8_t>(i,j) == 1 || rangeMat.at<float>(i,j) == FLT_MAX){
                    labelMat.at<int>(i,j) = -1;
                }
            }
        }

        // 如果有节点订阅groundCloud，那么就需要把地面点发布出来
		// 具体实现过程：把点放到groundCloud队列中去
        if (pubGroundCloud.getNumSubscribers() != 0){
            for (size_t i = 0; i <= groundScanInd; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (groundMat.at<int8_t>(i,j) == 1)
                        groundCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                }
            }
        }
    }

    void cloudSegmentation(){
        // segmentation process
        for (size_t i = 0; i < N_SCAN; ++i)
            for (size_t j = 0; j < Horizon_SCAN; ++j)
                // 如果labelMat[i][j]=0,表示没有对该点进行过分类
				// 需要对该点进行聚类
                if (labelMat.at<int>(i,j) == 0)
                    labelComponents(i, j);

        int sizeOfSegCloud = 0;
        // extract segmented cloud for lidar odometry
        for (size_t i = 0; i < N_SCAN; ++i) {
            // segMsg.startRingIndex[i]
			// segMsg.endRingIndex[i]
			// 表示第i线的点云起始序列和终止序列
			// 以开始线后的第5点为开始，以结束线前的第6点为结束
            segMsg.startRingIndex[i] = sizeOfSegCloud-1 + 5;

            for (size_t j = 0; j < Horizon_SCAN; ++j) {
                if (labelMat.at<int>(i,j) > 0 || groundMat.at<int8_t>(i,j) == 1){   // 找到可用的特征点或者地面点(不选择labelMat[i][j]=0的点)
                    // outliers that will not be used for optimization (always continue)
                    if (labelMat.at<int>(i,j) == 999999){  // labelMat数值为999999表示这个点是因为聚类数量不够30而被舍弃的点
                        if (i > groundScanInd && j % 5 == 0){  // 当列数为5的倍数，并且行数较大，可以认为非地面点的，将它保存进异常点云(界外点云)中
                            outlierCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]); 
                            continue;      //然后再跳过本次循环
                        }else{
                            continue;  // 需要舍弃的点直接continue跳过本次循环
                        }
                    }
                    // majority of ground points are skipped  
                    if (groundMat.at<int8_t>(i,j) == 1){  // 如果是地面点,对于列数不为5的倍数的，直接跳过不处理
                        if (j%5!=0 && j>5 && j<Horizon_SCAN-5) 
                            continue;
                    }
                    // 上面多个if语句已经去掉了不符合条件的点，这部分直接进行信息的拷贝和保存操作
                    // mark ground points so they will not be considered as edge features later
                    segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i,j) == 1); // true - ground point, false - other points，初始值为false
                    // mark the points' column index for marking occlusion later
                    segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;
                    // save range info
                    segMsg.segmentedCloudRange[sizeOfSegCloud]  = rangeMat.at<float>(i,j);
                    // save seg cloud
                    segmentedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of seg cloud
                    ++sizeOfSegCloud;
                }
            }

            segMsg.endRingIndex[i] = sizeOfSegCloud-1 - 5;
        }
        
        // extract segmented cloud for visualization
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            for (size_t i = 0; i < N_SCAN; ++i){
                for (size_t j = 0; j < Horizon_SCAN; ++j){
                    if (labelMat.at<int>(i,j) > 0 && labelMat.at<int>(i,j) != 999999){
                        segmentedCloudPure->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                        segmentedCloudPure->points.back().intensity = labelMat.at<int>(i,j);  //强度为点云聚类的标签值
                    }
                }
            }
        }
    }

    //*点聚类函数，基于BFS的点云聚类
    void labelComponents(int row, int col){  
        // use std::queue std::vector std::deque will slow the program down greatly
        float d1, d2, alpha, angle;
        int fromIndX, fromIndY, thisIndX, thisIndY; 
        bool lineCountFlag[N_SCAN] = {false};

        //queueStartInd，queueEndInd两个变量，类似于双指针。queueStartInd用于遍历，queueEndInd用于插入满足聚类条件，新的点索引
        queueIndX[0] = row;
        queueIndY[0] = col;
        int queueSize = 1;
        int queueStartInd = 0;           
        int queueEndInd = 1;

        allPushedIndX[0] = row;
        allPushedIndY[0] = col;
        int allPushedIndSize = 1;
        
        // 标准的BFS
        // BFS的作用是以(row，col)为中心向外面扩散，
        // 判断(row,col)是否是这个平面中一点
        while(queueSize > 0){
            // Pop point
            fromIndX = queueIndX[queueStartInd];
            fromIndY = queueIndY[queueStartInd];
            --queueSize;
            ++queueStartInd;  //逐个遍历顶点(变量1)
            // Mark popped point
            labelMat.at<int>(fromIndX, fromIndY) = labelCount;  // labelCount的初始值为1，后面会递增

            // Loop through all the neighboring grids of popped grid
            // neighbor=[[-1,0];[0,1];[0,-1];[1,0]]
			//*遍历点[fromIndX,fromIndY]边上的四个邻点
            for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter){
                // new index
                thisIndX = fromIndX + (*iter).first;
                thisIndY = fromIndY + (*iter).second;
                // index should be within the boundary(行索引要有边界0～15)
                if (thisIndX < 0 || thisIndX >= N_SCAN)
                    continue;
                // at range image margin (left or right side)
               // 是个环状的图片，左右连通
                if (thisIndY < 0)
                    thisIndY = Horizon_SCAN - 1;
                if (thisIndY >= Horizon_SCAN)
                    thisIndY = 0;

                // prevent infinite loop (caused by put already examined point back)
                // 如果点[thisIndX,thisIndY]已经标记过
				// labelMat中，-1代表无效点，0代表未进行标记过，其余为其他的标记
				//*如果当前的邻点已经标记过，则跳过该点。
				// 如果labelMat已经标记为正整数，则已经聚类完成，不需要再次对该点聚类
                if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                    continue;

                d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY),    //相邻点中距离lidar中心的最大者
                              rangeMat.at<float>(thisIndX, thisIndY));
                d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY),    //相邻点中距离lidar中心的最小者
                              rangeMat.at<float>(thisIndX, thisIndY));

                // alpha代表角度分辨率，
				// X方向上角度分辨率是segmentAlphaX(rad)
				// Y方向上角度分辨率是segmentAlphaY(rad)
                if ((*iter).first == 0)     //同一行相邻点(列相邻)
                    alpha = segmentAlphaX;    // segmentAlphaX = 0.2 / 180.0 * M_PI;
                else                                 //同一列相邻点(行相邻)
                    alpha = segmentAlphaY;   //segmentAlphaY = 2.0 / 180.0 * M_PI;

                // 通过下面的公式计算这两点之间是否有平面特征
				//*atan2(y,x)的值越大，d1，d2之间的差距越小,越平坦
                angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));

                if (angle > segmentTheta){
                    // segmentTheta=1.0472<==>60度
					// 如果算出角度大于60度，则假设这是个平面
                    queueIndX[queueEndInd] = thisIndX;
                    queueIndY[queueEndInd] = thisIndY;
                    ++queueSize;
                    ++queueEndInd;  //末尾插入点(变量2)

                    labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                    lineCountFlag[thisIndX] = true;

                    allPushedIndX[allPushedIndSize] = thisIndX;
                    allPushedIndY[allPushedIndSize] = thisIndY;
                    ++allPushedIndSize;     //统计聚类点的个数
                }
            }
        }

        // check if this segment is valid
        bool feasibleSegment = false;
        if (allPushedIndSize >= 30)          // 如果聚类超过30个点，直接标记为一个可用聚类，labelCount需要递增          （可能是面特帧聚类)
            feasibleSegment = true;
        else if (allPushedIndSize >= segmentValidPointNum){      // 如果聚类点数小于30大于等于5，统计竖直方向上的聚类点数
            int lineCount = 0;
            for (size_t i = 0; i < N_SCAN; ++i)
                if (lineCountFlag[i] == true)
                    ++lineCount;
            if (lineCount >= segmentValidLineNum)  //竖直方向上超过3个也将它标记为有效聚类                       (可能是线特征聚类)
                feasibleSegment = true;            
        }
        // segment is valid, mark these points
        if (feasibleSegment == true){
            ++labelCount;
        }else{ // segment is invalid, mark these points
            for (size_t i = 0; i < allPushedIndSize; ++i){
                labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;  //标记为999999的是需要舍弃的聚类的点，因为他们的数量小于30个
            }
        }
    }

    
    void publishCloud(){
        // 1. Publish Seg Cloud Info
        // 发布cloud_msgs::cloud_info消息
        segMsg.header = cloudHeader;
        pubSegmentedCloudInfo.publish(segMsg);

        // 2. Publish clouds
        sensor_msgs::PointCloud2 laserCloudTemp;
        // pubOutlierCloud发布界外点云
        pcl::toROSMsg(*outlierCloud, laserCloudTemp);
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubOutlierCloud.publish(laserCloudTemp);

        // segmented cloud with ground
        //发布带有地面的分割点云
        pcl::toROSMsg(*segmentedCloud, laserCloudTemp);      
        laserCloudTemp.header.stamp = cloudHeader.stamp;
        laserCloudTemp.header.frame_id = "base_link";
        pubSegmentedCloud.publish(laserCloudTemp);

        // projected full cloud（发布全部的点云)
        if (pubFullCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullCloud, laserCloudTemp);  //fullCloud中是带有图模型矩阵索引的点云，不带有点距离lidar中心的距离
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullCloud.publish(laserCloudTemp);
        }

        // original dense ground cloud（地面点)
        if (pubGroundCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*groundCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubGroundCloud.publish(laserCloudTemp);
        }
        // segmented cloud without ground(发布不带有地面的分割点云)
        if (pubSegmentedCloudPure.getNumSubscribers() != 0){
            pcl::toROSMsg(*segmentedCloudPure, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubSegmentedCloudPure.publish(laserCloudTemp);
        }
        // projected full cloud info（发布全部的点云，点的强度值为点距离lidar中心的距离)
        if (pubFullInfoCloud.getNumSubscribers() != 0){
            pcl::toROSMsg(*fullInfoCloud, laserCloudTemp);
            laserCloudTemp.header.stamp = cloudHeader.stamp;
            laserCloudTemp.header.frame_id = "base_link";
            pubFullInfoCloud.publish(laserCloudTemp);
        }
    }
};



//!主函数
int main(int argc, char** argv){

    ros::init(argc, argv, "lego_loam");
    
    ImageProjection IP; //创建类对象

    ROS_INFO("\033[1;32m---->\033[0m Image Projection Started.");

    ros::spin();
    return 0;
}
