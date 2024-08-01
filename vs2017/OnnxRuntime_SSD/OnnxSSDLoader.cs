using OpenCvSharp;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

static class OnnxSSD
{
    static string[] cocoLabels = new string[] { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear","hair drier", "toothbrush" };

    static public string RunSession(string onnxFilePath, string imageFilePath, int inputWidth = 1200, int inputHeight = 1200, bool ImShow = false)
    {
        Mat imgSrc = Cv2.ImRead(imageFilePath, ImreadModes.Color);
        string dstString = RunSessionAndDrawMat(onnxFilePath, imgSrc, inputWidth, inputHeight, ImShow);
        imgSrc.Dispose();

        return dstString;
    }

    static public string RunSessionAndDrawMat(string onnxFilePath, Mat imgSrc, int inputWidth = 1200, int inputHeight = 1200, bool ImShow = false)
    {
        using (var session = new InferenceSession(onnxFilePath))
        {
            return RunSessionAndDrawMat(session, imgSrc, inputWidth, inputHeight, ImShow);
        }
    }

    static public string RunSessionAndDrawMat(InferenceSession session, Mat imgSrc, int inputWidth = 1200, int inputHeight = 1200, bool ImShow = false)
    {
        var input = getDenseTensorFromMat(imgSrc, inputWidth, inputHeight);

        var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("image", input)
                };

        List<string> LineOutput = new List<string>();

        using (var results = session.Run(inputs))
        {
            Tensor<float> boxes = results.First().AsTensor<float>();
            Tensor<long> labels = results.ElementAt(1).AsTensor<long>();
            Tensor<float> scores = results.ElementAt(2).AsTensor<float>();

            // Process results
            for (int i = 0; i < scores.Length; i++)
            {
                if (scores[0,i] > 0.5) // Confidence threshold
                {
                    var label = cocoLabels[labels[0, i] - 1]; // Labels are 1-based
                    var p1_x = boxes[0, i, 0] * imgSrc.Width;
                    var p1_y = boxes[0, i, 1] * imgSrc.Height;
                    var p2_x = boxes[0, i, 2] * imgSrc.Width;
                    var p2_y = boxes[0, i, 3] * imgSrc.Height;

                    DrawRectangleAndLabel(imgSrc, p1_x, p1_y, p2_x, p2_y, label);
                    LineOutput.Add(label + "\t" + scores[0,i].ToString("g4") + "\t" + p1_x.ToString("0.0") + "\t" + p1_y.ToString("0.0") + "\t" + p2_x.ToString("0.0") + "\t" + p2_y.ToString("0.0"));
                }
            }
        }

        if (ImShow)
        {
            Cv2.ImShow("Image", imgSrc);
            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();
        }

        return string.Join("\r\n", LineOutput.ToArray());
    }

    static private DenseTensor<float> getDenseTensorFromMat(Mat src, int tensorWidth, int tensorHeight)
    {
        var dstTensor = new DenseTensor<float>(new[] { 1, 3, tensorHeight, tensorWidth });

        OpenCvSharp.Size newSize = new OpenCvSharp.Size(tensorWidth, tensorHeight);
        Mat dst = new Mat();
        Cv2.Resize(src, dst, newSize);

        float[] meanVec = { 0.485f, 0.456f, 0.406f };
        float[] stddevVec = { 0.229f, 0.224f, 0.225f };

        for (int y = 0; y < tensorHeight; y++)
        {
            for (int x = 0; x < tensorWidth; x++)
            {
                Vec3b color = dst.At<Vec3b>(y, x);
                dstTensor[0, 0, y, x] = ((float)color.Item2 / 255f - meanVec[0]) / stddevVec[0];
                dstTensor[0, 1, y, x] = ((float)color.Item1 / 255f - meanVec[1]) / stddevVec[1];
                dstTensor[0, 2, y, x] = ((float)color.Item0 / 255f - meanVec[2]) / stddevVec[2];
            }
        }

        dst.Dispose();
        return dstTensor;
    }

    static private void DrawRectangleAndLabel(Mat src, float p1_x, float p1_y, float p2_x, float p2_y, string label)
    {
        OpenCvSharp.Point p1 = new OpenCvSharp.Point(p1_x, p1_y);
        OpenCvSharp.Point p2 = new OpenCvSharp.Point(p2_x, p2_y);

        Cv2.Rectangle(src, p1, p2, Scalar.Red, thickness: 2);
        Cv2.PutText(src, label, p1, HersheyFonts.HersheySimplex, 1.0, Scalar.LightSkyBlue, thickness: 2);
    }
}