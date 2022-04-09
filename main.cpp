#include <CLI/CLI.hpp>
#include <logging/Checks.h>
#include <nlohmann/json.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;
namespace xf2d = cv::xfeatures2d;

const int HESSIAN = 400;
const int NUMMATCH = 20;
const float FLANNTHRES = 0.7;
const float RANSACCONF = 0.999;
const float RANSACTHRES = 0.6;
const float DISTTHRES = 0.2;

cv::Mat getImage(fs::path image_path) {
  std::cout << "\nReading: " << image_path.string() << std::endl;
  XR_CHECK_TRUE(exists(image_path), "Error: File not exist!");

  cv::Mat im = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
  XR_CHECK_FALSE(im.empty(), "Error: Fail to open file!");
  XR_CHECK_EQ(1, im.channels(), "Error: Fail to convert to grayscale!");
  std::cout << "Success: Image size: " << im.rows << " x " << im.cols << std::endl;

  return im;
}

cv::Mat getDepth(fs::path depth_path) {
  std::cout << "\nReading: " << depth_path.string() << std::endl;
  XR_CHECK_TRUE(exists(depth_path), "Error: File not exist!");

  cv::Mat depth = cv::imread(depth_path, cv::IMREAD_UNCHANGED);
  XR_CHECK_FALSE(depth.empty(), "Error: Fail to open file!");
  XR_CHECK_EQ(1, depth.channels(), "Error: Depth should have only 1 channel!");
  std::cout << "Success: Depth size: " << depth.rows << " x " << depth.cols << std::endl;

  return depth;
}

cv::Mat getIntrinsic(fs::path calib_path, const float width, const float height) {
  std::cout << "\nReading: " << calib_path.string() << std::endl;
  XR_CHECK_TRUE(exists(calib_path), "Error: File not exist!");

  std::ifstream calibStream(calib_path);
  nlohmann::json calibParams;
  calibStream >> calibParams;

  // Intrinsic properties
  // [ fx   0  cx
  //    0  fy  cy
  //    0   0   1 ]
  cv::Mat intrinsic(3, 3, CV_32FC1, 0.0);
  try {
    intrinsic.at<float>(0, 2) = (float)calibParams["cx"] * width;
    intrinsic.at<float>(1, 2) = (float)calibParams["cy"] * height;
    intrinsic.at<float>(0, 0) = (float)calibParams["fx"] * width;
    intrinsic.at<float>(1, 1) = (float)calibParams["fy"] * height;
    intrinsic.at<float>(2, 2) = 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: Please check cx, cy, fx, cy all exist in file!" << std::endl;
    exit(1);
  }

  std::cout << "Success: Intrinsic:" << std::endl;
  std::cout << intrinsic << std::endl;
  return intrinsic;
}

void ensureRatio(const cv::Mat& image, cv::Mat& depth) {
  float ratio = (float)image.rows / (float)image.cols;
  if ((float)depth.rows / (float)depth.cols == ratio) {
    // Follows height-width ratio
    std::cout << "Pass: valid depth size" << std::endl;
  } else if ((float)depth.cols / (float)depth.rows == ratio) {
    // Follows width-height ratio
    std::cout
        << "Pass: depth follows the image height-width ratio but with different orientation,\n"
        << "      rotating 90 degrees clockwise." << std::endl;
    cv::rotate(depth, depth, cv::ROTATE_90_CLOCKWISE);
  } else {
    std::cerr << "Error: depth does not follow image height-width ratio!" << std::endl;
    exit(1);
  }
  cv::resize(depth, depth, image.size());
}

void getCorrespondences(
    const cv::Mat& obj_im,
    const cv::Mat& scene_im,
    const cv::Mat& obj_depth,
    const cv::Mat& scene_depth,
    std::vector<cv::Point2i>& obj,
    std::vector<cv::Point2i>& scene) {
  std::cout << "\nPerforming SURF KNN matching." << std::endl;
  // Create Speeded Up Robust Features descriptor
  cv::Ptr<xf2d::SURF> detector = xf2d::SURF::create(HESSIAN);
  std::vector<cv::KeyPoint> keypoints_obj, keypoints_scene;
  cv::Mat descriptors_obj, descriptors_scene;
  detector->detectAndCompute(obj_im, cv::noArray(), keypoints_obj, descriptors_obj);
  detector->detectAndCompute(scene_im, cv::noArray(), keypoints_scene, descriptors_scene);

  // Fast Library for Approximate Nearest Neighbors matching
  cv::Ptr<cv::DescriptorMatcher> matcher =
      cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  std::vector<std::vector<cv::DMatch>> matches;
  matcher->knnMatch(descriptors_obj, descriptors_scene, matches, 2);

  // Lowe's match filtering
  std::vector<cv::DMatch> good_matches;
  for (int i = 0; i < matches.size(); i++) {
    if (matches[i][0].distance < FLANNTHRES * matches[i][1].distance) {
      int objX = keypoints_obj[matches[i][0].queryIdx].pt.x;
      int objY = keypoints_obj[matches[i][0].queryIdx].pt.y;
      int sceneX = keypoints_scene[matches[i][0].trainIdx].pt.x;
      int sceneY = keypoints_scene[matches[i][0].trainIdx].pt.y;
      if (std::abs(obj_depth.at<float>(objX, objY) - scene_depth.at<float>(sceneX, sceneY)) <
          DISTTHRES) {
        // Find target object in the query scene for matching
        good_matches.emplace_back(matches[i][0]); // Debug purpose
        obj.emplace_back(cv::Point2i(objX, objY));
        scene.emplace_back(cv::Point2i(sceneX, sceneY));
      }
    }
  }

  if (obj.size() < NUMMATCH) {
    std::cerr << "Error: Query image can't be matched to the target!" << std::endl;
    std::cerr << "       Insufficient correspondences: " << obj.size() << std::endl;
    exit(1);
  } else {
    std::cout << "Found number of correspondences: " << obj.size() << std::endl;
    return;
  }

  /*
  // Debug purpose
  cv::Mat img_matches;
  drawMatches(
      obj_im,
      keypoints_obj,
      scene_im,
      keypoints_scene,
      good_matches,
      img_matches,
      cv::Scalar::all(-1),
      cv::Scalar::all(-1),
      std::vector<char>(),
      cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
  resize(img_matches, img_matches, cv::Size(256, 384));
  imshow("Matches", img_matches);
  cv::waitKey();
  */
}

int main(int argc, char* argv[]) {
  // Parse inputs
  CLI::App app{"LavaMap Image Match Test by Leying Hu"};
  std::string cwd;
  app.add_option("--cwd", cwd, "Directory that saves target and query data.")
      ->required()
      ->check(CLI::ExistingDirectory);
  std::string target;
  app.add_option("--target", target, "Name of target jpeg, depth tiff, and calib json files.")
      ->required();
  std::string query;
  app.add_option("--query", query, "Name of query jpeg, depth tiff, and calib json files.")
      ->required();

  CLI11_PARSE(app, argc, argv);

  // Load inputs
  std::cout << "\n=================Parsing Target=================" << std::endl;
  cv::Mat target_image = getImage(fs::path(cwd) / (target + ".jpeg"));
  cv::Mat target_depth = getDepth(fs::path(cwd) / (target + ".depth.tiff"));
  cv::Mat target_intrinsic =
      getIntrinsic(fs::path(cwd) / (target + ".json"), target_image.cols, target_image.rows);

  std::cout << "\n=================Parsing Query==================" << std::endl;
  cv::Mat query_image = getImage(fs::path(cwd) / (query + ".jpeg"));
  cv::Mat query_depth = getDepth(fs::path(cwd) / (query + ".depth.tiff"));
  cv::Mat query_intrinsic =
      getIntrinsic(fs::path(cwd) / (query + ".json"), query_image.cols, query_image.rows);

  // Check image and depth sizes
  // Jake: assume images have the same size and same orientation
  if (target_image.rows == query_image.rows && target_image.cols == query_image.cols) {
    std::cout << "Pass: target image size and query image size are the same." << std::endl;
  } else {
    std::cerr << "Error: target image size and query image size are different!" << std::endl;
    return 1;
  }

  std::cout << "\n================Checking Validity===============" << std::endl;
  // Jake: Depth images can be rotated 90 degrees clockwise to match the color images
  std::cout << "\nChecking target depth validity." << std::endl;
  ensureRatio(target_image, target_depth);
  std::cout << "\nChecking query depth validity." << std::endl;
  ensureRatio(target_image, query_depth);

  std::cout << "\n===========Matching Target and Query============" << std::endl;
  // Detect correspondences
  // Find target object in the query scene for matching
  std::vector<cv::Point2i> target_pt, query_pt;
  getCorrespondences(target_image, query_image, target_depth, query_depth, target_pt, query_pt);

  std::cout << "\n============Computing Transformation============" << std::endl;
  // Finding Essential Matrix and its Rotation and Translation Matrix components
  cv::Mat essentialMat, rotationMat, translationMat, maskMat, tri_pt;
  std::cout << "\nUsing 5-point algorithm to estimate Essential Matrix..." << std::endl;
  essentialMat = cv::findEssentialMat(
      query_pt, target_pt, query_intrinsic, cv::RANSAC, RANSACCONF, RANSACTHRES, maskMat);
  // Save maskMat binaries because it will be
  // reset after parsing to cv::recoverPose()
  std::cout << "\nEssential Matrix:\n" << essentialMat << std::endl;
  std::cout << "\nDecomposing Rotation and Translation from Essential Matrix..." << std::endl;
  cv::recoverPose(
      essentialMat,
      query_pt,
      target_pt,
      query_intrinsic,
      rotationMat,
      translationMat,
      RANSACTHRES,
      maskMat,
      tri_pt);
  std::cout << "\nRotation Matrix:\n" << rotationMat << std::endl;
  std::cout << "\nTranslation Matrix:\n" << translationMat << std::endl;

  // Concatenating Extrinsic Matrix
  cv::Mat extrinsicMat = cv::Mat(4, 4, CV_32FC1, cv::Scalar::all(0));
  rotationMat.copyTo(extrinsicMat.rowRange(0, 3).colRange(0, 3));
  translationMat.copyTo(extrinsicMat.rowRange(0, 3).col(3));
  extrinsicMat.at<float>(3, 3) = 1;
  std::cout << "\nConcatenated Extrinsic Matrix:\n" << extrinsicMat << std::endl;

  std::cout << "\n============Warping Query To Target=============" << std::endl;
  // Find homography matrix for image warping from query image to target image
  std::cout << "\nUsing RANSAC to estimate Homography Matrix..." << std::endl;
  cv::Mat homographyMat = cv::findHomography(query_pt, target_pt, cv::RANSAC);
  std::cout << "\nFound Homography Matrix:\n" << homographyMat << std::endl;
  std::cout << "\nPerspective transforming the query into the target image..." << std::endl;

  std::cout << "\n===================Displaying===================" << std::endl;
  std::cout << "\nPress any key to save the displaying warped query image!" << std::endl;
  cv::Mat source = cv::imread(fs::path(cwd) / (query + ".jpeg"), cv::IMREAD_UNCHANGED);
  cv::Mat reference = cv::imread(fs::path(cwd) / (target + ".jpeg"), cv::IMREAD_UNCHANGED);
  cv::Mat warped_image;
  cv::warpPerspective(source, warped_image, homographyMat, reference.size());
  // // debug purpose
  // addWeighted(reference, 0.2, warpped_image, 0.8, 0.0, warpped_image);
  imshow("Warpped Query Image", warped_image);
  cv::waitKey();

  std::string output_image = (fs::path(cwd) / (query + "_warp.jpeg")).string();
  imwrite(output_image, warped_image);
  std::cout << "\nWarped image saved to " << output_image << std::endl;
  return 0;
}
