/*
 * Utility.hpp
 *
 *  Created on: Oct 26, 2012
 *      Author: xuchi
 */

#ifndef UTILITY_HPP_
#define UTILITY_HPP_

#include <unistd.h>
#include <stdio.h>
#include <vector>
#include <string.h>

#define Utility_Check(X) \
	if (X) ; else \
	{printf("Utility_Check(%s) : %s : line %d\n",#X,__FILE__,__LINE__); exit(-1);}

//! Deal with file
/*! The usage of the class is as follows:
 \code
 int main()
 {
 XFILE file = fopen("tmp.txt", "r");
 printf("test getch()\n");
 while (true)
 {
 char ch;
 int n = file.getch(ch);
 if (!n)
 {
 break;
 }
 printf("%c", ch);
 }

 fseek(file.pf, 0, SEEK_SET);
 printf("test getline()\n");
 char line[1024];
 while (true)
 {
 printf("### line ### ");
 int n = file.getline(line, 1024);
 if (!n)
 {
 break;
 }
 printf("%s", line);
 }
 file.close();
 return 1;
 }
 \endcode
 */
struct XFILE
{
	static const int xfile_len_buf = 1024;

	FILE *pf;
	char buf[xfile_len_buf]; //!< buffer for getch()
	int ibuf; //!< the index in buffer
	int nbuf; //!< the number of elements in buffer

	XFILE() :
			ibuf(0), nbuf(0)
	{
		buf[0] = 0;
	}

	XFILE(FILE *pf_) :
			ibuf(0), nbuf(0)
	{
		pf = pf_;
	}

	int getch(char &ch)
	{
		if (nbuf <= 0)
		{
			loadbuf: nbuf = fread(buf, 1, xfile_len_buf, pf);
			if (nbuf <= 0)
			{
				nbuf = 0;
				return 0;
			}
			ibuf = 0;
		}
		if (ibuf < nbuf)
		{
			ch = buf[ibuf];
			ibuf++;
		}
		else
		{
			goto loadbuf;
		}
		return 1;
	}

	int getline(char line[], int maxlen)
	{
		int i = 0;
		while (i < maxlen)
		{
			char ch;
			int res = getch(ch);
			if (res)
			{
				line[i] = ch;
				i++;
				if (ch == '\n')
				{
					line[i] = 0;
					break;
				}
			}
			else
			{
				line[i] = 0;
				break;
			}
		}
		return i;
	}

	void close()
	{
		fclose(pf);
	}
};

//! Useful utility functions
struct Utility_
{
	//! check if a file exists
	static int exist(const char *filename)
	{
		FILE* f = 0;
		f = fopen(filename, "r");
		int flag = bool(f);
		if (f)
		{
			fclose(f);
		}
		return flag;
	}

	//! sleep for millisecond
	static void msleep(int ms)
	{
		usleep(ms * 1000);
	}
}; 

#endif /* UTILITY_HPP_ */
