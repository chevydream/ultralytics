from ultralytics import YOLO
import cv2

runControl = 6      # 运行模式：0-


# 读取USB摄像头的实时数据进行推理
def myVideoPredict(model):
    # 打开USB摄像头
    camera_nu = 0  # 摄像头编号
    cap = cv2.VideoCapture(camera_nu + cv2.CAP_DSHOW)

    while cap.isOpened():
        # 获取图像
        res, frame = cap.read()
        # 如果读取成功
        if res:
            # 正向推理
            results = model(frame)
            # 绘制结果
            annotated_frame = results[0].plot()
            # 显示图像
            cv2.imshow(winname="YOLOV8", mat=annotated_frame)
            # 按ESC退出
            if cv2.waitKey(1) == 27:
                break
    # 释放连接
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    srcs = [
        'E:/DLCode/视频/0000542-0500052.mp4',
        'E:/DLCode/视频/0000336-0500753.mp4',
    ]

    model = YOLO("E:/DLModel/流量检测/yolov8n.pt")

    # 推理2：视频文件推理(内存不增长的）
    if runControl == 2:
        results = model.predict(srcs[1], classes=[2,5,6,7], stream=True) #预测
        # results = model.track(srcs[0], classes=[2,5,6,7], stream=True) #跟踪
        cv2.namedWindow('test', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        for result in results:
            res_plotted = result.plot()
            cv2.imshow('test', res_plotted)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()

    # 推理3：视频文件推理(内存一直增长）
    if runControl == 3:
        results = model.predict(source="E:/DLCode/视频/0000542-0500052.mp4")

    # 推理4：读取USB摄像头的实时数据进行推理
    if runControl == 4:  # 方法1
        results = model.predict(source=0+cv2.CAP_DSHOW, show=True)

    # 推理5：读取USB摄像头的实时数据进行推理
    if runControl == 5:  # 方法2
        myVideoPredict(model)

    # 推理6：读取rtsp格式的实时视频进行推理
    if runControl == 6:
        results = model.predict(source="rtsp://192.168.2.74:8557/h264", show=True)




