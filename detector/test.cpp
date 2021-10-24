#include "Detector.h"
#include <ctime>
#include<numeric>
#include <algorithm>


cv::Rect get_my_rect(cv::Mat& img, float bbox[4]) {
	int l, r, t, b;
	float r_w = Yolo::INPUT_W / (img.cols * 1.0);
	float r_h = Yolo::INPUT_H / (img.rows * 1.0);
	if (r_h > r_w) {
		l = bbox[0] - bbox[2] / 2.f;
		r = bbox[0] + bbox[2] / 2.f;
		t = bbox[1] - bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
		b = bbox[1] + bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
		l = l / r_w;
		r = r / r_w;
		t = t / r_w;
		b = b / r_w;
	}
	else {
		l = bbox[0] - bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
		r = bbox[0] + bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
		t = bbox[1] - bbox[3] / 2.f;
		b = bbox[1] + bbox[3] / 2.f;
		l = l / r_h;
		r = r / r_h;
		t = t / r_h;
		b = b / r_h;
	}
	return cv::Rect(l, t, r - l, b - t);
}


bool rectA_intersect_rectB(cv::Rect rectA, cv::Rect rectB)
{
	if (rectA.x > rectB.x + rectB.width) { return false; }
	if (rectA.y > rectB.y + rectB.height) { return false; }
	if ((rectA.x + rectA.width) < rectB.x) { return false; }
	if ((rectA.y + rectA.height) < rectB.y) { return false; }

	float colInt = std::min(rectA.x + rectA.width, rectB.x + rectB.width) - std::max(rectA.x, rectB.x);
	float rowInt = std::min(rectA.y + rectA.height, rectB.y + rectB.height) - std::max(rectA.y, rectB.y);
	float intersection = colInt * rowInt;
	float areaA = rectA.width * rectA.height;
	float areaB = rectB.width * rectB.height;
	float intersectionPercent = intersection / (areaA + areaB - intersection);

	if ((0 < intersectionPercent) && (intersectionPercent < 1) && (intersection != areaA) && (intersection != areaB))
	{
		return true;
	}

	return false;
}

//判断rect1是否在rect2里面
bool isInside(cv::Rect rect1, cv::Rect rect2)
{
	return (rect1 == (rect1 & rect2));
}

//计算两个框的iou
float DecideOverlap(const cv::Rect& r1, const cv::Rect& r2)
{
	int x1 = r1.x;
	int y1 = r1.y;
	int width1 = r1.width;
	int height1 = r1.height;

	int x2 = r2.x;
	int y2 = r2.y;
	int width2 = r2.width;
	int height2 = r2.height;

	int endx = std::max(x1 + width1, x2 + width2);
	int startx = std::min(x1, x2);
	int width = width1 + width2 - (endx - startx);

	int endy = std::max(y1 + height1, y2 + height2);
	int starty = std::min(y1, y2);
	int height = height1 + height2 - (endy - starty);

	float ratio = 0.0f;
	float Area, Area1, Area2;

	if (width <= 0 || height <= 0)
		return 0.0f;
	else
	{
		Area = width * height;
		Area1 = width1 * height1;
		Area2 = width2 * height2;
		ratio = Area / (Area1 + Area2 - Area);
	}

	return ratio;
}

//std::vector<Yolo::Detection> post_filter(cv::Mat frame, std::vector<Yolo::Detection> pre_object, std::vector<Yolo::Detection> object) {
//	std::vector<Yolo::Detection> out_object(object);
//	
//
//	for (auto it : pre_object) {
//		out_object.push_back(it);
//	}
//	return out_object;
//
//}
//剔除远小于其他轮子的轮子
std::vector<Yolo::Detection> removeTOOsmall(cv::Mat frame, std::vector<Yolo::Detection> object) {
	std::vector<int> area_rec;
	std::cout << frame.rows << std::endl;
	std::cout << frame.cols << std::endl;


	//int x = frame.cols * 0.15;
	//int y = 0;
	//int width = frame.cols * 0.85;
	//int height = frame.rows;
	//cv::Rect crop(x, y, width, height);

	//cv::Mat tmp = frame(crop);
	//cv::imshow("crop", tmp);
	//cv::waitKey();
	for (auto i : object) {
		cv::Rect r = get_my_rect(frame, i.bbox);
		int area = r.area();
		std::cout << "area" << " " << area << std::endl;

		//cv::Rect inter = r & crop;
		//if (inter.area() > 0)
		area_rec.push_back(area);
	}
	double sum = std::accumulate(area_rec.begin(), area_rec.end(), 0.0);
	double mean = sum / area_rec.size(); //均值
	std::cout << "均值" << mean << std::endl;

	//double accum = 0.0;
	//std::for_each(area_rec.begin(), area_rec.end(), [&](const double d) {
	//	accum += (d - mean) * (d - mean);
	//	});
	//double stdev = sqrt(accum / (area_rec.size() - 1)); //方差
	//std::cout << "方差" << stdev << std::endl;

	int gate = 0.6 * mean;
	std::cout << "gate" << gate << std::endl;

	std::vector<Yolo::Detection> res;
	for (int i = 0; i < area_rec.size(); ++i) {
		if (area_rec[i] < gate)
			continue;
		res.push_back(object[i]);
	}

	return res;
}

std::vector<Yolo::Detection> post_filter(cv::Mat frame, std::vector<Yolo::Detection> pre_object, std::vector<Yolo::Detection> object) {
	std::vector<int> area_rec;

	for (auto i : pre_object) {
		cv::Rect r = get_my_rect(frame, i.bbox);
		int area = r.area();
		//std::cout << "area" << " " << area << std::endl;
		area_rec.push_back(area);
	}
	auto maxPosition = std::max_element(area_rec.begin(), area_rec.end());
	int i = maxPosition - area_rec.begin();


	std::vector<Yolo::Detection> pre_object1 = { pre_object[i] };

	cv::Rect raw = get_my_rect(frame, pre_object1[0].bbox);

	std::vector<Yolo::Detection> out_object;

	//排除object里面不在pre_object1[0]框中的
	for (int i = 0; i < object.size(); ++i) {
		cv::Rect r = get_my_rect(frame, object[i].bbox);
		///*if (rectA_intersect_rectB(raw, r))
		//	continue;*/
		//if (isInside(r, raw) || rectA_intersect_rectB(raw, r))
		//	continue;
		//else
		//{
		//	object.erase(object.begin() + i);

		//}
		//std::cout << DecideOverlap(r, raw) << std::endl;

		if (DecideOverlap(r, raw) != 0) {
			out_object.push_back(object[i]);
		}

	}
	//std::cout << object.size() << std::endl;
	//out_object = removeTOOsmall(frame, out_object);
	out_object.push_back(pre_object1[0]);

	return out_object;

}


//移除一个框同时与多个框相交的框
std::vector<Yolo::Detection> remove_inter(cv::Mat frame, std::vector<Yolo::Detection> object) {
	std::vector<Yolo::Detection> res;
	for (int i = 0; i < object.size(); ++i) {
		cv::Rect r1 = get_my_rect(frame, object[i].bbox);
		int count = 0;
		for (int j = 0; j < object.size(); ++j) {
			if (i != j) {
				cv::Rect r2 = get_my_rect(frame, object[j].bbox);
				//std::cout << DecideOverlap(r1, r2) << " ";
				if (DecideOverlap(r1, r2) > 0.1)
					count++;
			}
		}
		//std::cout << count;
		//std::cout << std::endl;
		if (count >= 2)
			continue;
		else
		{
			res.push_back(object[i]);
		}
	}
	return res;

}




//若一个框在另一个框的内部，则移除他
//std::vector<Yolo::Detection> remove_small(cv::Mat frame, std::vector<Yolo::Detection> object) {
//	std::vector<Yolo::Detection> res;
//	for (int i = 0; i < object.size(); ++i) {
//		cv::Rect r1 = get_my_rect(frame, object[i].bbox);
//		cv::Rect r2;
//		for (int j = 0; j < object.size(); ++j) {
//			r2 = get_my_rect(frame, object[j].bbox);
//		}
//		if (isInside(r1, r2)) {
//			continue;
//		}
//		else {
//			res.push_back(object[i]);
//		}
//
//	}
//
//	return res;
//
//
//}


void SavePicture(std::string addr, cv::Mat frame, int nums) {
	std::string text = "count: " + std::to_string(nums);

	int font_face = cv::FONT_HERSHEY_COMPLEX;
	double font_scale = 1;
	int thickness = 2;
	int baseline;
	cv::Point origin;
	origin.x = 5;
	origin.y = 30;
	cv::putText(frame, text, origin, font_face, font_scale, cv::Scalar(0, 255, 0), thickness, 8, 0);

	auto dot_index = addr.find_last_of(".");
	std::string pre_addr_result = addr.substr(0, dot_index);
	std::string post_addr_result = addr.substr(dot_index, addr.size());

	//addr_result.erase(dot_index, static_cast<int>(addr_result.end() - dot_index));
	std::string addr_result = pre_addr_result + "_result" + post_addr_result;

	cv::imwrite(addr_result, frame);
}

bool parse_args(int argc, char** argv, std::string& addr) {
	if (argc < 1) return false;
	if (std::string(argv[1]) == "-a" && (argc == 3)) {
		addr = std::string(argv[2]);
	}
	else {
		return false;
	}
	return true;
}


int main(int argc, char** argv)
{
	std::string addr = "";
	if (!parse_args(argc, argv, addr)) {
		std::cerr << "arguments not right!" << std::endl;
		return -1;
	}
	cv::Mat frame = cv::imread(addr);

	std::vector<Yolo::Detection> pre_object;

	{
		Detector PreDetect("./model/pre.engine");
		pre_object = PreDetect.Detect(frame);
	}
	

	int pre_nums = pre_object.size();
	//std::cout << "number of trucks : " << pre_nums << std::endl;


	Detector MyDetect("./model/best.engine");
	//cv::Mat frame = cv::imread(addr);
	auto object_old = MyDetect.Detect(frame);

	//去除同时和其他(>=2)框iou大于一定值的框
	//auto object_without_inter = remove_inter(frame, object_old);
	auto object = remove_inter(frame, object_old);

	//若一个框在另一个框的内部，则移除他
	//auto object = remove_small(frame, object_without_inter);

	//object = eliminate_false(frame, object);
	//如果第一步检测到车了，若没有检测到车就pass
	std::vector<Yolo::Detection> out_object;
	int nums;
	if (pre_nums > 0) {
		out_object = post_filter(frame, pre_object, object);
		nums = out_object.size() - 1;

	}
	else {
		out_object = object;
		nums = out_object.size();
	}

	std::cout << "number of wheels : " << nums << std::endl;

	MyDetect.DrawRectangle(frame, out_object);

	SavePicture(addr, frame, nums);
	
	return 0;
}
//	/*
//	int cam;
//	std::cin >> cam;
//	Detector MyDetect("./model/best.engine");
//	//Detector MyDetect("C:\\Users\\yangy\\Desktop\\yolov5\\model\\best.engine");
//	cv::VideoCapture capture;
//	capture.open(cam);
//
//	cv::Mat frame;
//	while (true)
//	{
//		if (!capture.read(frame))
//			continue;
//		auto tag1 = clock();
//		auto object = MyDetect.Detect(frame);
//		std::cout << "number of wheels : " << object.size() << std::endl;
//		auto tag2 = clock();
//		std::cout << tag2 - tag1 << "ms" << std::endl;
//		MyDetect.DrawRectangle(frame, object);
//		cv::imshow("ok", frame);
//		cv::waitKey(1);
//	}
//	*/
//	
//	
//	
//}