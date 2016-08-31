#ifndef __TinyOgre_h_
#define __TinyOgre_h_

#include <OgreRoot.h>
#include <OgreCamera.h>
#include <OgreSceneManager.h>
#include <OgreRenderWindow.h>
#include <OgreLogManager.h>
#include <OgreViewport.h>
#include <OgreConfigFile.h>
#include <OgreEntity.h>
#include <OgreWindowEventUtilities.h>
#include <OgreHardwarePixelBuffer.h>
#include <OgreTextureManager.h>
#include <OgreTexture.h>
#include <OgreGLTexture.h>
#include <OgreRenderSystem.h>

#include <string>
#include <GL/glew.h>
#include <GL/freeglut.h>
 
struct HandRenderer2
{
	Ogre::Root *mRoot;
	Ogre::Camera* mCamera;
	Ogre::SceneManager* mSceneMgr;
	Ogre::RenderWindow* mWindow;
	Ogre::Entity* handEntity;
	Ogre::SceneNode* handNode;
	Ogre::TexturePtr rtt_texture;
	Ogre::RenderTexture *renderTexture;
	Ogre::ResourceGroupManager *resource_mgr_;

	GLuint glTextureID;

	const std::string render_tex_rsrc_name_;
	const std::string render_tex_name_;

	HandRenderer2(void) :
			mRoot(0), mCamera(0), mSceneMgr(0), mWindow(0), render_tex_rsrc_name_("HandRenderer RenderTexture Resources"), render_tex_name_("HandRenderer RenderTexture")
	{
	}

	virtual ~HandRenderer2(void)
	{
		delete mRoot;
	}

	bool Init(void)
	{
		mRoot = new Ogre::Root("plugins.cfg");

		Ogre::ConfigFile cf;
		cf.load("resources.cfg");
		Ogre::ConfigFile::SectionIterator seci = cf.getSectionIterator();
		Ogre::String secName, typeName, archName;
		while (seci.hasMoreElements())
		{
			secName = seci.peekNextKey();
			Ogre::ConfigFile::SettingsMultiMap *settings = seci.getNext();
			Ogre::ConfigFile::SettingsMultiMap::iterator i;
			for (i = settings->begin(); i != settings->end(); ++i)
			{
				typeName = i->first;
				archName = i->second;
				Ogre::ResourceGroupManager::getSingleton().addResourceLocation(
						archName, typeName, secName);
			}
		}
		if (mRoot->restoreConfig() || mRoot->showConfigDialog())
		{
			// If returned true, user clicked OK so initialise
			// Here we choose to let the system create a default rendering window by passing 'true'
			mWindow = mRoot->initialise(true, "Hand Render Window");
		}
		else
		{
			return false;
		}

		//-------------------------------------------------------------------------------------
		// choose scenemanager
		// Get the SceneManager, in this case a generic one
		mSceneMgr = mRoot->createSceneManager(Ogre::ST_GENERIC);
		mCamera = mSceneMgr->createCamera("HandCamera");

		mCamera->setPosition(Ogre::Vector3(0, 0, 0));
		mCamera->lookAt(Ogre::Vector3(0, 0, -1));
		mCamera->setNearClipDistance(5);
		Ogre::Viewport* vp = mWindow->addViewport(mCamera);
		vp->setBackgroundColour(Ogre::ColourValue(0, 0, 0));
		mCamera->setAspectRatio(
				Ogre::Real(vp->getActualWidth())
						/ Ogre::Real(vp->getActualHeight()));

		//-------------------------------------------------------------------------------------
		// Set default mipmap level (NB some APIs ignore this)
//		Ogre::TextureManager::getSingleton().setDefaultNumMipmaps(5);
//		Ogre::ResourceGroupManager::getSingleton().initialiseAllResourceGroups();

		resource_mgr_ = Ogre::ResourceGroupManager::getSingletonPtr();
		resource_mgr_->createResourceGroup(render_tex_rsrc_name_);
		resource_mgr_->initialiseAllResourceGroups();

		rtt_texture =
				Ogre::TextureManager::getSingleton().createManual(render_tex_name_, render_tex_rsrc_name_ , Ogre::TEX_TYPE_2D, mWindow->getWidth(), mWindow->getHeight(), 0, Ogre::PF_R8G8B8A8, Ogre::TU_RENDERTARGET, 0, false, false);

		renderTexture = rtt_texture->getBuffer()->getRenderTarget();

		renderTexture->addViewport(mCamera);
		renderTexture->getViewport(0)->setClearEveryFrame(true);
		//renderTexture->getViewport(0)->setBackgroundColour(Ogre::ColourValue::Black);
		//renderTexture->getViewport(0)->setOverlaysEnabled(false);

		handEntity = mSceneMgr->createEntity("Hand", "hand.mesh");
		handNode = mSceneMgr->getRootSceneNode()->createChildSceneNode();
		handNode->attachObject(handEntity);
		handNode->setPosition(0, 0, -700);
		handNode->setScale(40, 40, 40);

		// Set ambient light
		mSceneMgr->setAmbientLight(Ogre::ColourValue(0.5, 0.5, 0.5));

		return true;
	}

	bool RenderOneFrame()
	{
        bool success = mRoot->renderOneFrame();
        renderTexture->update();

        Ogre::WindowEventUtilities::messagePump();

		if (mWindow->isClosed())
		{
			return false;
		}

		// Render a frame
		return success;
	}

	GLuint getTextureID()
	{
		Ogre::GLTexture *p = (Ogre::GLTexture*)rtt_texture.get();
		return p->getGLID();
	}
};

#endif // #ifndef __TinyOgre_h_
