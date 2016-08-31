#include "Hand_Match.h"
#include "HandPose2.hpp"
#include <vector>
#include "CudaHostTimer.h"

const int image_width = 128 ;
const int image_height = 128 ;

const int d1 = 500;
const int d2 = 2000;

struct Reg
{
	float val[26];
};

void getRefParams(int N, std::vector<Reg> &vecReg, Reg &input) ;

int main()
{

	HandPose2_::HandPose2 handpose ;
	handpose.hand_renderer.mCamera->setNearClipDistance(d1) ;
	handpose.hand_renderer.mCamera->setFarClipDistance(d2) ;

	const int N = 200;

    Hand_Match hand_match( handpose, image_width , image_height, N );

	std::vector<Reg> vecReg;
	Reg input;
	getRefParams(N, vecReg, input);

	std::vector< std::vector < float > > hypothesis(N) ;
	std::vector< float > reference( input.val, input.val + 26 ) ;

	//assert( reference.size() == 26 ) ;

	for ( int i = 0; i < N ; i++ )
	{
		hypothesis[i].assign( vecReg[i].val, vecReg[i].val + 26 ) ;
		//assert( hypothesis[i].size() == 26 ) ;
	}

	CudaHostTimer timer;
	timer.start();

    hand_match.set_Reference_Image( reference ) ;

    hand_match.set_Hypothesis_Images( hypothesis ) ;

    printf( " \n Max Score: %i \n", hand_match.find_Top_N_Matches( ) );

    timer.stop();
    printf( "Elapsed Time: %f\n" , timer.value() );

    return 0;

}

void getRefParams(int N, std::vector<Reg> &vecReg, Reg &input)
{
	using namespace std;

	vector<vector<float> > fingerParams;
	for (int i = 0; i < 10; i++)
	{
		char buf[20];
		sprintf(buf, "pose/%d.pose", i);

		std::vector<float> param(20, 0);
		std::ifstream fin(buf);
		for (int j = 0; j < 20; j++)
		{
			fin >> param[j];
		}
		fingerParams.push_back(param);
	}

	vecReg.resize(N, Reg());
	for (int i = 0; i < N; i++)
	{
		///////ask Chris about correct order of parameters /////////////

		vecReg[i].val[23] = rand() % 50 - 25;
		vecReg[i].val[24] = rand() % 50 - 25;
		vecReg[i].val[25] = rand() % 100 - 50 - 750;
		vecReg[i].val[20] = rand() % 30 - 15;
		vecReg[i].val[21] = rand() % 60 - 30 + 90;
		vecReg[i].val[22] = rand() % 60 - 30 - 90;

		int idx = rand() % 10;
		for (int j = 0; j < 20; j++)
		{
			vecReg[i].val[j] = fingerParams[idx][j];
		}
	}

	input = vecReg[rand() % N];
	input.val[0] += rand() % 3;
	input.val[1] += rand() % 3;
	input.val[2] += rand() % 3;
	input.val[3] += rand() % 3;
	input.val[4] += rand() % 3;
	input.val[5] += rand() % 3;
}

