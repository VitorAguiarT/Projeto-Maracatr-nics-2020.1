//Projeto Maracatrônics nº 2 - RECONHECIMENTO FACIAL

//Inicialização de todas as bibliotecas necessárias para o bom funcionamento da aplicação:
// --------------------------------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>

using namespace std;

#include <opencv2/core.hpp>
#include <opencv2/face.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/videoio.hpp>

using namespace cv;
using namespace cv::face;
// --------------------------------------------------------------------------------------

//Inicialização de variáveis globais como por exemplo, variável para captura vídeo, contador, modelo de reconhecimento facial e classificador de fazes...
// --------------------------------------------------------------------------------------
CascadeClassifier face_cascade;

Ptr<FaceRecognizer> model =  LBPHFaceRecognizer::create();

VideoCapture capture;

int numberFaces = 0, id[50], contador = 0, numberPhotos = 0;
string fileName[50000];
bool captureAux = false;
// --------------------------------------------------------------------------------------

/*Inicialização das funções criadas para a aplicação:
  => int photosCapture(Mat frame, int id, int contador); -> Função que tira imagens dos rostos dos usuários a
  partir da câmera principal do notebook e armazenam em uma pasta ("Banco de Dados").
  
  => void photosRecognition(Mat frame); -> Função que lê as imagens do Banco de Dados, detecta os rostos a partir 
  da câmera principal do notebook e identifica se um rosto está presente ou não nos arquivos salvos e se sim, qual 
  usuário foi encontrado.
*/
// --------------------------------------------------------------------------------------
int photosCapture(Mat frame, int id, int contador);
void photosRecognition(Mat frame);
// --------------------------------------------------------------------------------------

int main() 
{
    //Inicialização das variáveis utilizadas na função main do programa:
    // --------------------------------------------------------------------------------------
    Mat frame;
    string aux;
    int i = 0, temp = 0, k = 0, opr = -1;
    bool ok = false, exit = false, training = false;
    // --------------------------------------------------------------------------------------

    //Leitura do arquivo base para o algoritmo de classificação de rostos:
    // --------------------------------------------------------------------------------------
    if(!face_cascade.load("haarcascade_frontalface_alt.xml")) 
    {
        cout << "--(!)Error loading face cascade" << endl;
        return -1;
    };
    // --------------------------------------------------------------------------------------

    //Loop principal do programa, onde podemos escolher a operação a ser executada:
    // --------------------------------------------------------------------------------------
    while(!exit) 
    {
        //OPERAÇÕES:
        // --------------------------------------------------------------------------------------
        if(opr == -1) 
        {
            //Interface principal do programa onde escolhemos a operação a ser executada:
            // --------------------------------------------------------------------------------------
            cout << "DIGITE A OPERACAO QUE DESEJA REALIZAR:" << endl;
            cout << "(0)- SAIR." << endl << "(1)- Cadastrar um rosto" << endl << "(2) Treinar para reconhecer os rostos presentes no banco de dados" << endl << "(3)- Reconhecer um rosto" << endl;
            cout << "(-1)- VOLTA PARA A ESCOLHA DA OPERACAO! " << "SEMPRE QUE QUISER VOLTAR A ESSA TELA, BASTA APERTAR A BARRA DE ESPACO" << endl;
            cin >> opr;
            // --------------------------------------------------------------------------------------
        } 
        
        //Operação 0 - Saída do programa:
        // --------------------------------------------------------------------------------------    
        else if(opr == 0) exit = true;
        // --------------------------------------------------------------------------------------
        
        else if(opr == 1) 
        {
            //Operação 1 - Cadastro de uma nova face para o Banco de Dados:
            // --------------------------------------------------------------------------------------
            ok = true;
            
            cout << "Digite um número para o seu id: ";
            cin >> id[numberFaces];
            
            capture.open(0);
            if(!capture.isOpened()) 
            {
                cout << "--(!)Error opening video capture" << endl;
                return -1;
            }

            cout << "--- POR FAVOR, MANTENHA O ROSTO VOLTADO PARA A CAMERA ---" << endl;
            while(capture.read(frame)) 
            {        
                if(frame.empty()) 
                {
                    cout << "--(!) No captured frame -- Break!" << endl;
                    break;
                }

                stringstream number1, number2;
                number1 << id[numberFaces];
                number2 << contador;
        
                fileName[numberPhotos] = "dataImages/User." + number1.str() + "." + number2.str() + ".jpg";

                contador = photosCapture (frame, id[numberFaces], contador);

                if(contador == 100) 
                {
                    cout << ">>> CAPTURA DE FACE COMPLETA! <<<" << endl;
                    captureAux = true;
                    break;
                }

                k = waitKey(1);
            }

            capture.release();
            destroyAllWindows();
            contador = 0;
            opr = -1;
            numberFaces++;
            // --------------------------------------------------------------------------------------
        }
        
        else if(opr == 2)
        {
            //Operação 2 - Treinamento do algorítmo de reconhecimento facial:
            // -------------------------------------------------------------------------------------- 
            //# ------------------------------------ TRAING THE RECOGNISER ----------------------------------------
            if(captureAux) 
            {
                if(!training) 
                {
                    cout << "AGUARDE......" << endl;
            
                    vector<Mat> images;
                    vector<int> labels;

                    cout << endl;
                    cout << "TRAINING......" << endl;

                    for(i = 0; i < numberFaces; i++) 
                    {
                        for(numberPhotos = temp; numberPhotos < ((i+1) * 100); numberPhotos++) 
                        {
                            images.push_back(imread(fileName[numberPhotos], 0));
                            labels.push_back(id[i]);
                            model->setLabelInfo(labels[id[i]], fileName[numberPhotos]);
                        }
                        temp = i*100;
                    }

                    cout << ".........." << endl;

                    model->train(images, labels);
                    training = true;
 
                    cout << "..............." << endl << "TREINAMENTO CONCLUIDO." << endl;
                } 
                
                else 
                {
                    cout << "AGUARDE......" << endl;
                    
                    vector<Mat> newimages;
                    vector<int> newlabels;

                    cout << endl;
                    cout << "TRAINING......" << endl;

                    for(i = 0; i < numberFaces; i++) 
                    {
                        for(numberPhotos = temp; numberPhotos < ((i+1) * 100); numberPhotos++) 
                        {
                            newimages.push_back(imread(fileName[numberPhotos], 0));
                            newlabels.push_back(id[i]);
                            model->setLabelInfo(newlabels[id[i]], fileName[numberPhotos]);
                        }
                        temp = i*100;
                    }

                    cout << ".........." << endl;

                    model->update(newimages, newlabels);
                    
                    cout << "..............." << endl << "TREINAMENTO CONCLUIDO." << endl;
                }
            } 

            else cout << "A CAPTURA NAO FOI CONCLUIDA COM SUCESSO, POR FAVOR, REINICIE O PROGRAMA E TENTE CAPTURAR AS FOTOS PARA O BANCO DE DADOS NOVAMENTE!" << endl;
            //# ------------------------------------ TRAING THE RECOGNISER ----------------------------------------
            opr = -1;
            // --------------------------------------------------------------------------------------
        } 
        
        else if(opr == 3)
        {
            // Operação 3 - Reconhecimento Facial:
            // --------------------------------------------------------------------------------------
            if(!ok && !training) 
            {
                opr = -1;
                cout << "CADASTRE UM ROSTO E EXECUTE O TREINO DO RECONHECIMENTO FACIAL PRIMEIRO!" << endl;
            } 
            
            else 
            { 
                capture.open(0);
                if(!capture.isOpened()) 
                {
                    cout << "--(!)Error opening video capture" << endl;
                    return -1;
                }

                while(capture.read (frame)) 
                {
                    if(frame.empty()) 
                    {     
                        cout << "--(!) No captured frame -- Break!" << endl;
                        break;
                    }
        
                    photosRecognition(frame);
            
                    k = waitKey(1);
                    if(k == 27) break; 

                    if(k == 32) 
                    {
                        opr = -1;
                        break;
                    }
                }

                capture.release();
                destroyAllWindows();
                opr = -1;
            }
            // --------------------------------------------------------------------------------------
        } 
        
        else 
        {
            //Operação NÃO EXISTENTE:
            // --------------------------------------------------------------------------------------
            cout << "OPERACAO INVALIDA!" << endl << "DIGITE NOVAMENTE:   " << endl;
            opr = -1;
            // --------------------------------------------------------------------------------------
        }
        // --------------------------------------------------------------------------------------
    }
    // --------------------------------------------------------------------------------------

    return 0;
}

int photosCapture(Mat frame, int id, int contador)
{    
    Mat face_resized, frame_gray;
    stringstream number1, number2;
    int lineType = LINE_8;
    
    number1 << id;
    number2 << contador;

    if(!frame.empty()) 
    {
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        equalizeHist(frame_gray, frame_gray);
    
        vector<Rect> faces;
        face_cascade.detectMultiScale(frame_gray, faces);   

        for(size_t i = 0; i < faces.size(); i++) 
        {
            if(faces.size() == 1) 
            {
                Point center_rectangle(faces[i].x, faces[i].y);
                rectangle(frame, center_rectangle, Size(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(255, 0, 0), 0, lineType);
                putText(frame, "Rosto Detectado:", Point(faces[i].x,faces[i].y-10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0));

                Rect face_i = faces[i];
                Mat face = frame_gray(face_i);
                
                face_resized = frame_gray;
                resize(face, face_resized, Size(100, 100), 1.0, 1.0, INTER_CUBIC);

                imshow("Capturando Fotos", frame);

                if(!face_resized.empty()) 
                {
                    imwrite("dataImages/User." + number1.str() + "." + number2.str() + ".jpg", face_resized);
                    imshow("Sistema de Captura de Faces para Reconhecimento Facial", face_resized);
                    contador++;
                    numberPhotos++;                
                }
            } 
            
            else if(faces.size() > 1) 
            {
                Point center_rectangle(faces[i].x, faces[i].y);
                rectangle(frame, center_rectangle, Size(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(0, 0, 255), 0, lineType);
                putText(frame, "Multiplos Rostos Detectados!", Point(faces[i].x,faces[i].y-10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255));
                putText(frame, "ERRO AO CAPTURAR ROSTO PARA O BANCO DE DADOS",Point (faces[i].x,faces[i].y+50), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,255));

                imshow("Capturando Fotos", frame);
            }
        }
    }

    return contador;
}

void photosRecognition(Mat frame) 
{
    Mat original = frame.clone ();
    Mat gray;
    string aux_id;
    string box_text;
    int lineType = LINE_8;
    
    cvtColor(original, gray, COLOR_BGR2GRAY);

    vector< Rect_<int> > faces;
    face_cascade.detectMultiScale(gray, faces);

    for(size_t i = 0; i < faces.size(); i++) 
    {
        Rect face_i = faces[i];
        Mat face = gray(face_i);

        Mat face_resized;
        resize(face, face_resized, Size(100, 100), 1.0, 1.0, INTER_CUBIC);

        int label = -1;
        double confidence = 0;
		
        int prediction = model->predict (face_resized);
        model->predict(face_resized, label, confidence);

        if(faces.size() >= 10) cout << "ALARME DE MULTIDÃO!" << endl;

        if(confidence > 100) 
        {
            box_text = format("Rosto desconhecido:");

            rectangle(original, face_i, Scalar(0, 0, 255), lineType);
            int pos_x = max(face_i.tl().x - 10, 0);
            int pos_y = max(face_i.tl().y - 10, 0);
            putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 2);
        } 
        
        else 
        {
            aux_id = model->getLabelInfo(prediction);
            box_text = format("id do Usuario: = %d", prediction);
            
            rectangle(original, face_i, Scalar(0, 255, 0), lineType);
            int pos_x = max(face_i.tl().x - 10, 0);
            int pos_y = max(face_i.tl().y - 10, 0);
            putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0), 2);
        }
    }

    imshow("RECONHECIMENTO FACIAL", original);
}
