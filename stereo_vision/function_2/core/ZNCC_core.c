#include <stdio.h>
#include <stdlib.h>
#include "include/DIC_comm.h"
#include "include/PSO_comm.h"


float ZNCC_cost_function(PSO_context *ctx, int *pos) {
	ctx->config.


}



/*ZNCC(zero-normalized cross-correlation)(-1 ~ +1) */
double DIC_ZNCC_cost(int Pi_u, int Pi_v, int Object_point[], int img_aft[][img_column],\
                     int img_aft_sub[][Size], int img_bef_sub[][Size], double Mean_bef[])   
{
	int i, j, Aft_sub_sum=0;
	int row, col;
	double Mean_aft, Sum_Numerator=0.0, Sum_Denominator_bef=0.0, Sum_Denominator_aft=0.0, coef=0.0;
	/* Construct img_aft_sub */
	for (i=0;i<Size;i++) 
	{
		for (j=0;j<Size;j++)
		{
			row = i - SizeHalf + Pi_u + Object_point[0];
			col = j - SizeHalf + Pi_v + Object_point[1];
			if(row>=img_row || row<0 || col>=img_column || col<0){
				printf("\n[ERROR] row>img_row || row<0 || col>img_column || col<img_column\n");
				exit(1);
			}
			img_aft_sub[i][j] = img_aft[row][col];
			if (img_aft_sub[i][j]>255 || img_aft_sub[i][j]<0){
				printf("\n[ERROR] img_aft_sub[i][j]>255 || img_aft_sub[i][j]<00");
				exit(1);
			}
		}
	}
	/* Mean of img_aft_sub */
	for (i=0;i<Size;i++)
	{
		for (j=0;j<Size;j++)
		{
			Aft_sub_sum+=img_aft_sub[i][j];
		}
	}
	Mean_aft=mean(Aft_sub_sum); /* mean function is defined by macro (#define)   */ 
	
	/* Substract its mean, comopute its sqrt and sum */
	for (i=0;i<Size;i++)
	{
		for (j=0;j<Size;j++)
		{
			Sum_Numerator+=(img_bef_sub[i][j] - Mean_bef[0])*(img_aft_sub[i][j] - Mean_aft);
			Sum_Denominator_bef+=square(img_bef_sub[i][j] - Mean_bef[0]);
			Sum_Denominator_aft+=square(img_aft_sub[i][j] - Mean_aft);
		}
	}
	if(Sum_Denominator_bef==0 || Sum_Denominator_aft==0){
		return 0.0;
	}
	coef = Sum_Numerator/(sqrt(Sum_Denominator_bef*Sum_Denominator_aft));
	return coef;
}