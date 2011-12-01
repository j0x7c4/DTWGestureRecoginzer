#include "DTWGestureRecognizer.h"
#include <stdio.h>

template <class T>
void convertFromString(T &value, const std::string &s) {
  std::stringstream ss(s);
  ss >> value;
}
void Split ( char* str, char* sep, vector<string>& token ) {
  char * p = strtok(str,sep);
  while ( p ) {
    token.push_back(string(p));
    p=strtok(NULL,sep);
  }
}
//Edit this function for different data format
int ReadData ( FILE* fp, vector<vector<double>>& seq ) {
  char tc[2];
  int dim,n;
  double t;
  if ( fscanf(fp,"%d,%d",&dim,&n)==EOF ) return 0;
  fgets(tc,2,fp);
  seq.clear();
  seq.resize(n,vector<double>());
  for ( int i=0 ; i<dim; i++ ){
    for ( int j=0 ; j<n ; j++ ) {
      fscanf(fp,"%lf,",&t);
      seq[j].push_back(t);
    }
  }
  return 1;
}

int ReadFile ( char* data_file_name, char* label , DTWGestureRecognizer& dtw ) { 
  int dim,m,n;
  double t;
  FILE *fp_data;
  char ts[200];
//parameters
  vector<string> tlabel;

  fp_data = fopen(data_file_name,"r");
  if ( !fp_data ) {
    printf("Open %s failed!\n",data_file_name);
    return 1;
  }

  //read data

  vector<vector<double>> tdata;
  while ( ReadData(fp_data,tdata) ) {
    dtw.Add(tdata,string(label));
  }
  fclose(fp_data);
  return 0;
}
//Training
void ClassicTraining ( char* list, DTWGestureRecognizer &dtw ) {
  FILE *fp;
  char data_file[100];
  char label[100];
  fp = fopen(list,"r");
  if ( !fp ) {
    fprintf(stderr,"Failed to open %s\n",list);
    exit(1);
  }
  printf("***********Training Models************\n");
  int data_size;
  while ( fscanf(fp,"%d",&data_size)!= EOF ) {
    fscanf(fp,"%s",label);
    for ( int i=0 ; i<data_size ; i++ ) {
      fscanf(fp,"%s",data_file);
      printf("Reading data %s...",data_file);
      if ( !ReadFile(data_file,label,dtw) ) {
        printf("Success!\n");
      }
      else {
        printf("Fail!\n");
      }
    }
  }
  fclose(fp);
}
void GaussianTraining ( char* list, DTWGestureRecognizer &dtw, char* label) {
  FILE *fp;
  char data_file[100];
  fp = fopen(list,"r");
  if ( !fp ) {
    fprintf(stderr,"Failed to open %s\n",list);
    exit(1);
  }
  printf("***********Training Models************\n");
  int data_size;
  fscanf(fp,"%d",&data_size);
  fscanf(fp,"%s",label);
  for ( int i=0 ; i<data_size ; i++ ) {
    fscanf(fp,"%s",data_file);
    printf("Reading data %s...",data_file);
    if ( !ReadFile(data_file,label,dtw) ) {
      printf("Success!\n");
    }
    else {
      printf("Fail!\n");
    }
  }
  fclose(fp);
  printf("Start to align sequences...");
  dtw.AliginSequence();
  printf("Done!\n");
  printf("Start to training label %s ",label);
  if ( !dtw.Training(label) ) printf("Success!\n");
  else printf("Fail!\n");
}

void ShowModel ( string label , DTWGestureRecognizer &dtw) {
  printf("***********Show Models************\n");
  printf("Label %s\n",label.c_str());
  dtw.PrintModel(label);
}
int main (int argc, char **argv) {
  FILE *fp;
  
  //parameters
  //-d dimension of feature vector
  //-th threshold
  //-w width of constraint window
  //-fin list file
  //-ft path of test file
  //-fs save path of model data file
  //-fout 
  int n;
  int dim=-1;
  int width=MAXINT;
  double threshold=0;
  char* fin_name=NULL;
  char* ft_name =NULL;
  char* fsave_name = NULL;
  char* fout_name = NULL;
  char* fm_list_name=NULL;

  for ( int i = 1 ; i<argc ; i+=2 ) {
    string p(argv[i]);
    char* value = argv[i+1];
    if ( p=="-th" ) {
      threshold = atof(value);
    }
    else if ( p=="-d" ) {
      dim=atoi(value);
    }
    else if ( p=="-w" ) {
      width=atoi(value);
    }
    else if ( p=="-fin" ) {
      fin_name = (char*)malloc(sizeof(char)*strlen(value));
      strcpy(fin_name,value);
    }
    else if ( p=="-ft" ) {
      ft_name = (char*)malloc(sizeof(char)*strlen(value));
      strcpy(ft_name,value);
    }
    else if ( p== "-fm" ) {
      fm_list_name = (char*)malloc(sizeof(char)*strlen(value));
      strcpy(fm_list_name,value);
    }
    else if ( p=="-fs" ) {
      fsave_name = (char*)malloc(sizeof(char)*strlen(value));
      strcpy(fsave_name,value);
    }
    else if ( p=="-fout" ) {
      fout_name = (char*)malloc(sizeof(char)*strlen(value));
      strcpy(fout_name,value);
    }
    else {
      printf("Wrong Parameters!\n");
      exit(1);
    }
  }
  
  if ( dim<0 ) {
    printf("We need a list file!\n");
    exit(1);
  }
  char label[100];
  DTWGestureRecognizer dtw(dim,width,threshold);
  if ( fin_name ) {
#ifdef GAUSSIAN
    GaussianTraining(fin_name,dtw,label);
#else
    ClassicTraining(fin_name,dtw);
#endif
  }
  if ( fm_list_name ) {
    fp = fopen(fm_list_name,"r");
    if ( !fp ) {
      fprintf(stderr,"Failed to open %s...\n",fm_list_name);
      exit(1);
    }
    char fm_name[100];
    while ( fscanf(fp,"%s",fm_name)!=EOF ) {
      FILE *fm_p = fopen(fm_name,"r");
      if ( !fm_p ) {
        fprintf(stderr,"Failed to open %s\n",fm_name);
        continue;
      }
      double v;
      fscanf(fm_p,"%s",label);
      fscanf(fm_p,"%d %d",&dim,&n);
      vector<vector<vector<double>>> covariances(n,vector<vector<double>>(dim,vector<double>(dim,0)));
      vector<vector<double>> means(n,vector<double>(dim,0));
      for ( int i=0 ; i<n ; i++ ) {
        for ( int j=0 ; j<dim ; j++ ) {
          for ( int k = 0  ; k<dim ; k++ ) {
            fscanf(fm_p,"%lf",&v);
            covariances[i][j][k]=v;
          }
        }
        for ( int j=0 ; j<dim ; j++ ) {
          fscanf(fm_p,"%lf",&v);
          means[i][j]=v;
        }
      }
      dtw.SetGaussianModel(n,dim,covariances,means,string(label));
      printf("Model %s loaded.\n",label);
      fclose(fm_p);
    }
    fclose(fp);
  }
  //vector<string> labels = dtw.GetUniqueLables();
#ifdef GAUSSIAN
  ShowModel(string(label),dtw);
#endif
  
  //save model
  if ( fsave_name ) {
    fp = fopen(fsave_name,"w");
    if ( !fp ) {
      fprintf(stderr,"Failed to open %s!\n",fsave_name);
    }
    fprintf(fp,"%s\n",label);
    dtw.Save(fp,label);
    printf("Saved!\n");
    fclose(fp);
  }

  //test
  time_t t_begin, t_end;
  if ( ft_name ) {

    printf("***********Testing data************\n");
    vector<vector<double>> test_seq;
    fp = fopen(ft_name,"r");
    if ( !fp ) {
      fprintf(stderr,"Failed to open %s!\n",ft_name);
    }
    else {
      char test_file[100];
      int k=1;
      while ( fscanf(fp,"%s",test_file)!=EOF ) {
        FILE *fp_test = fopen(test_file,"r");
        if ( !fp_test ) {
          fprintf(stderr,"Failed to open %s!\n",test_file);
          continue;
        }
        int t=1;
        while ( ReadData(fp_test,test_seq) ) {
          printf("(%d) Testing data %d in %s...",k++,t++,test_file);
          double p;
          string res;
          t_begin = time(NULL);
#ifdef GAUSSIAN
          res = dtw.RecognizeByGaussianModel(test_seq,p);
#else 
          res = dtw.Recognize(test_seq,p);
#endif
          t_end = time(NULL);
          printf("%lf\n",p);
          printf(" %s, time=%dms\n",res.c_str(),t_end-t_begin);
        }
        fclose(fp_test);
      }
    }
    fclose(fp);
  }

  return 0;
}