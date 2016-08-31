#ifndef HANDPOSE_HPP_
#define HANDPOSE_HPP_

#include "HandRender2.hpp"
#include <OGRE/OgreManualObject.h>
#include <fstream>
#include <GL/gl.h>
#include <vector>
#include "Utility.hpp"
#include <opencv2/opencv.hpp>

namespace HandPose2_
{
using namespace Ogre;
using namespace std;

struct HandPose2
{
	HandRenderer2 hand_renderer;

	std::vector<int> idxFinger;
	int nJoint;

	SceneNode *root;
	SceneNode* hand_node;
	Entity* hand_entity;
	SceneManager *sceneMgr;
	Skeleton* skeleton;

	std::vector<ManualObject*> vec_axis;
	std::vector<SceneNode*> vec_node;
	std::vector<Quaternion> vec_jointBase;
	std::vector<Quaternion> vec_jointRotate;

	IplImage *depth;
	IplImage *rgb;
	IplImage *shortdepth;

	ManualObject *depthmap;

	HandPose2(int width = 640, int height = 480) :
			depthmap(0)
	{
		int indFinger_[] =
		{ 7, 11, 15, 19, 3 };
		for (int i = 0; i < sizeof(indFinger_) / sizeof(int); i++)
		{
			idxFinger.push_back(indFinger_[i]);
		}

		// init render
		hand_renderer.Init();
		depth = cvCreateImage(cvSize(width, height), 32, 1);
		rgb = cvCreateImage(cvSize(width, height), 8, 3);
		shortdepth = cvCreateImage(cvSize(width, height), 16, 1);

		// init camera
		hand_renderer.mCamera->setFOVy(Degree(43));
		hand_renderer.mCamera->setPosition(0, 0, 0);
		hand_renderer.mCamera->lookAt(0, 0, -1000);

		// init hand_node
		hand_node = hand_renderer.handNode;
		Utility_Check(hand_node);

		sceneMgr = hand_renderer.mSceneMgr;
		root = sceneMgr->getRootSceneNode();
		hand_entity = hand_renderer.handEntity;
		Utility_Check(hand_entity);
		Utility_Check(hand_entity->hasSkeleton());
		skeleton = hand_entity->getSkeleton();
		nJoint = skeleton->getNumBones();
		for (int i = 0; i < nJoint; i++)
		{
			// set manually operate
			skeleton->getBone(i)->setManuallyControlled(true);
		}

		// init axis
		float sz_axis = 100;
		for (int i = 0; i < nJoint + 1; i++)
		{
			char buf[128];
			sprintf(buf, "axis%d", i);
			vec_axis.push_back(sceneMgr->createManualObject(buf));

			vec_axis[i]->begin("BaseWhiteNoLighting",
					RenderOperation::OT_LINE_LIST);
			vec_axis[i]->position(0, 0, 0);
			vec_axis[i]->colour(1, 0, 0, 1);
			vec_axis[i]->position(sz_axis, 0, 0);
			vec_axis[i]->colour(1, 0, 0, 1);
			vec_axis[i]->position(0, 0, 0);
			vec_axis[i]->colour(0, 1, 0, 1);
			vec_axis[i]->position(0, sz_axis, 0);
			vec_axis[i]->colour(0, 1, 0, 1);
			vec_axis[i]->position(0, 0, 0);
			vec_axis[i]->colour(0, 0, 1, 1);
			vec_axis[i]->position(0, 0, sz_axis);
			vec_axis[i]->colour(0, 0, 1, 1);
			vec_axis[i]->end();

			sprintf(buf, "node%d", i);
			vec_node.push_back(root->createChildSceneNode(buf));
			vec_node[i]->attachObject(vec_axis[i]);
			vec_node[i]->setVisible(false);
		}

		// init jointBase and jointRotate
		for (int i = 0; i < nJoint; i++)
		{
			Bone* bone = skeleton->getBone(i);
			Utility_Check(bone);
			vec_jointBase.push_back(bone->getOrientation());
			vec_jointRotate.push_back(Quaternion::IDENTITY);
		}
	}

	~HandPose2()
	{
		if (shortdepth)
		{
			cvReleaseImage(&shortdepth);
		}
		if (depth)
		{
			cvReleaseImage(&depth);
		}
		if (rgb)
		{
			cvReleaseImage(&rgb);
		}
	}

	void Render()
	{
		hand_renderer.RenderOneFrame();
//		glReadPixels(0, 0, depth->width, depth->height, GL_DEPTH_COMPONENT,
//				GL_FLOAT, depth->imageData);
//		glReadPixels(0, 0, rgb->width, rgb->height, GL_BGR, GL_UNSIGNED_BYTE,
//				rgb->imageData);
//		cvFlip(rgb, rgb, 0);
//		cvFlip(depth, depth, 0);
//		float d1 = hand_renderer.mCamera->getNearClipDistance();
//		float d2 = hand_renderer.mCamera->getFarClipDistance();
//		cvConvertScale(depth, shortdepth, d2 - d1, d1);
	}

	void ShowJoints(int i)
	{
		Bone* bone = skeleton->getBone(i);

		Vector3 a = bone->convertLocalToWorldPosition(Vector3::ZERO);
		a = hand_node->convertLocalToWorldPosition(a);
		cout << a << endl;

		Quaternion b = bone->convertLocalToWorldOrientation(
				Quaternion::IDENTITY);

		vec_node[i]->setPosition(a);
		vec_node[i]->setOrientation(b);
		vec_node[i]->setVisible(true);
	}

	void HideJoints()
	{
		for (int i = 0; i < nJoint; i++)
		{
			vec_node[i]->setVisible(false);
		}
	}

	void SetPose(std::vector<float> &param)
	{
		for (int i = 0; i < 5; i++)
		{
			float a = param[i * 4];
			float b1 = param[i * 4 + 1];
			float b2 = param[i * 4 + 2];
			float b3 = param[i * 4 + 3];

			int idx = idxFinger[i];
			Matrix3 mat;

			mat.FromEulerAnglesXYZ(Degree(b1), Degree(0), Degree(a));
			vec_jointRotate[idx].FromRotationMatrix(mat);
			skeleton->getBone(idx)->setOrientation(
					vec_jointBase[idx] * vec_jointRotate[idx]);

			idx++;
			mat.FromEulerAnglesXYZ(Degree(b2), Degree(0), Degree(0));
			vec_jointRotate[idx].FromRotationMatrix(mat);
			skeleton->getBone(idx)->setOrientation(
					vec_jointBase[idx] * vec_jointRotate[idx]);

			idx++;
			mat.FromEulerAnglesXYZ(Degree(b3), Degree(0), Degree(0));
			vec_jointRotate[idx].FromRotationMatrix(mat);
			skeleton->getBone(idx)->setOrientation(
					vec_jointBase[idx] * vec_jointRotate[idx]);
		}
	}

	std::vector<float> GetParam(const char *filename)
	{
		std::vector<float> param;
		ifstream f(filename);
		for (int i = 0; i < 5; i++)
		{
			char buf[512];
			f.getline(buf, 512);
			istringstream s(buf);
			float a, b1, b2, b3;
			s >> a >> b1 >> b2 >> b3;
			param.push_back(a);
			param.push_back(b1);
			param.push_back(b2);
			param.push_back(b3);
		}
		return param;
	}

	void ReadParam(const char *filename)
	{
		ifstream f(filename);
		for (int i = 0; i < 5; i++)
		{
			char buf[512];
			f.getline(buf, 512);
			istringstream s(buf);
			float a, b1, b2, b3;
			s >> a >> b1 >> b2 >> b3;
			cout << a << "," << b1 << "," << b2 << "," << b3 << endl;

			int idx = idxFinger[i];
			Matrix3 mat;

			mat.FromEulerAnglesXYZ(Degree(b1), Degree(0), Degree(a));
			vec_jointRotate[idx].FromRotationMatrix(mat);
			skeleton->getBone(idx)->setOrientation(
					vec_jointBase[idx] * vec_jointRotate[idx]);

			idx++;
			mat.FromEulerAnglesXYZ(Degree(b2), Degree(0), Degree(0));
			vec_jointRotate[idx].FromRotationMatrix(mat);
			skeleton->getBone(idx)->setOrientation(
					vec_jointBase[idx] * vec_jointRotate[idx]);

			idx++;
			mat.FromEulerAnglesXYZ(Degree(b3), Degree(0), Degree(0));
			vec_jointRotate[idx].FromRotationMatrix(mat);
			skeleton->getBone(idx)->setOrientation(
					vec_jointBase[idx] * vec_jointRotate[idx]);
		}
	}

	void WriteParam(const char *filename)
	{
		ofstream f(filename);
		Radian x, y, z;
		Matrix3 mat;
		for (int i = 0; i < 5; i++)
		{
			int idx = idxFinger[i];

			vec_jointRotate[idx].ToRotationMatrix(mat);
			mat.ToEulerAnglesXYZ(x, y, z);
			f << Degree(z).valueDegrees() << "\t" << Degree(x).valueDegrees()
					<< "\t";

			vec_jointRotate[idx + 1].ToRotationMatrix(mat);
			mat.ToEulerAnglesXYZ(x, y, z);
			f << Degree(x).valueDegrees() << "\t";

			vec_jointRotate[idx + 2].ToRotationMatrix(mat);
			mat.ToEulerAnglesXYZ(x, y, z);
			f << Degree(x).valueDegrees() << endl;
		}
	}

	//! mix two poses by weight
	static std::vector<float> MixParam(std::vector<float> &a,
			std::vector<float> &b, float w)
	{
		int n = a.size();
		std::vector<float> c(n);
		Utility_Check(a.size() == b.size());
		for (int i = 0; i < n; i++)
		{
			c[i] = a[i] * (1 - w) + b[i] * w;
		}
		return c;
	}

	//! mix two poses by a list of weights
	static std::vector<float> MixParam(std::vector<float> &a,
			std::vector<float> &b, std::vector<float> &w)
	{
		int n = a.size();
		std::vector<float> c(n);
		Utility_Check(a.size() == b.size());
		Utility_Check(a.size() == w.size());
		for (int i = 0; i < n; i++)
		{
			c[i] = a[i] * (1 - w[i]) + b[i] * w[i];
		}
		return c;
	}

	GLuint getTextureID()
	{
		return hand_renderer.getTextureID();
	}
};

inline Quaternion EulerXYZ(Radian x, Radian y, Radian z)
{
	Matrix3 mat;
	mat.FromEulerAnglesXYZ(x, y, z);
	Quaternion rot;
	rot.FromRotationMatrix(mat);
	return rot;
}
}
#endif /* HANDPOSE_HPP_ */
