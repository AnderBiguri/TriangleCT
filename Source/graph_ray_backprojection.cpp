
#include "graph_ray_projection.hpp"


// This flag activates timing of the code
#define DEBUG_TIME 1
#define EPSILON 0.000001


void testKernel(const Graph * graph,float * d_res){
    d_res[0]=(float)graph->node[1].position[0];
    
};
/*********************************************************************
 *********************** Cross product in CUDA ************************
 ********************************************************************/
vec3d cross(const vec3d a,const vec3d b)
{
    vec3d c;
    c.x= a.y*b.z - a.z*b.y;
    c.y= a.z*b.x - a.x*b.z;
    c.z= a.x*b.y - a.y*b.x;
    return c;
}
/*********************************************************************
 *********************** Dot product in CUDA ************************
 ********************************************************************/
double dot(const vec3d a, const vec3d b)
{
    
    return a.x*b.x+a.y*b.y+a.z*b.z;
}



float max4(float *t,int* indM){
    float max=0;
    *indM=-1;
    for(int i=0;i<4;i++){
        if (t[i]>max){
            max=t[i];
            *indM=i;
        }
    }
    return max;
}

float min4nz(float *t){
    float min=1;
    for(int i=0;i<4;i++)
        min=(t[i]<min && t[i]!=0)?t[i]:min;
        return min;
}

int nnz(float *t){
    int nz=0;
    for(int i=0;i<4;i++){
        if(t[i]>0){
            nz++;
        }
    }
    return nz;
    
}
/*********************************************************************
 *********************** Moller trumbore ************************
 ********************************************************************/
float moller_trumbore(const vec3d ray1, const vec3d ray2,
        const vec3d trip1,const vec3d trip2,const vec3d trip3, const float safetyEpsilon){
    
    
    vec3d direction,e1,e2;
    
    direction.x=ray2.x-ray1.x;     direction.y=ray2.y-ray1.y;     direction.z=ray2.z-ray1.z;
    e1.x       =trip2.x-trip1.x;   e1.y       =trip2.y-trip1.y;   e1.z       =trip2.z-trip1.z;
    e2.x       =trip3.x-trip1.x;   e2.y       =trip3.y-trip1.y;   e2.z       =trip3.z-trip1.z;
    
    
    vec3d q=cross(direction,e2);
    double a=dot(e1,q);
    
    if ((a>-EPSILON) & (a<EPSILON)){
        // the vector is parallel to the plane (the intersection is at infinity)
        return 0.0f;
    }
    
    double f=1/a;
    vec3d s;
    s.x=ray1.x-trip1.x;     s.y=ray1.y-trip1.y;     s.z=ray1.z-trip1.z;
    double u=f*dot(s,q);
    
    if (u<0.0-safetyEpsilon){
        // the intersection is outside of the triangle
        return 0.0f;
    }
    
    vec3d r=cross(s,e1);
    double v= f*dot(direction,r);
    
    if (v<0.0-safetyEpsilon || (u+v)>1.0+safetyEpsilon){
        // the intersection is outside of the triangle
        return 0.0;
    }
//     mexPrintf("%.16f %.16f %.16f\n",q.x,q.y,q.z);
//     mexPrintf("%.16f  %.16f %.16f %.16f %.16f\n",a,f,u,v,f*dot(e2,r));
    return f*dot(e2,r);
    
    
    
}

/*********************************************************************
 **********************Tetra-line intersection************************
 ********************************************************************/

// TODO: check if adding if-clauses after each moller trumbore is better of worse.
bool tetraLineIntersect(const unsigned long *elements,const double *vertices,
        const vec3d ray1, const vec3d ray2,
        const unsigned long elementId,float *t,bool computelenght,const float safetyEpsilon){
    
    unsigned long auxNodeId[4];
    auxNodeId[0]=elements[elementId*4+0];
    auxNodeId[1]=elements[elementId*4+1];
    auxNodeId[2]=elements[elementId*4+2];
    auxNodeId[3]=elements[elementId*4+3];
    
    
    vec3d triN1,triN2,triN3;
    
    float l1,l2,l3,l4;
    
    ///////////////////////////////////////////////////////////////////////
    // As modular arithmetic is bad on GPUs (flop-wise), I manually unroll the loop
    //for (int i=0;i<4;i++)
    ///////////////////////////////////////////////////////////////////////
    // Triangle
    triN1.x=vertices[auxNodeId[0]*3+0];    triN1.y=vertices[auxNodeId[0]*3+1];    triN1.z=vertices[auxNodeId[0]*3+2];
    triN2.x=vertices[auxNodeId[1]*3+0];    triN2.y=vertices[auxNodeId[1]*3+1];    triN2.z=vertices[auxNodeId[1]*3+2];
    triN3.x=vertices[auxNodeId[2]*3+0];    triN3.y=vertices[auxNodeId[2]*3+1];    triN3.z=vertices[auxNodeId[2]*3+2];
    //compute
    l1=moller_trumbore(ray1,ray2,triN1,triN2,triN3,safetyEpsilon);
    // Triangle
    triN1.x=vertices[auxNodeId[0]*3+0];    triN1.y=vertices[auxNodeId[0]*3+1];    triN1.z=vertices[auxNodeId[0]*3+2];
    triN2.x=vertices[auxNodeId[1]*3+0];    triN2.y=vertices[auxNodeId[1]*3+1];    triN2.z=vertices[auxNodeId[1]*3+2];
    triN3.x=vertices[auxNodeId[3]*3+0];    triN3.y=vertices[auxNodeId[3]*3+1];    triN3.z=vertices[auxNodeId[3]*3+2];
    //compute
    l2=moller_trumbore(ray1,ray2,triN1,triN2,triN3,safetyEpsilon);
    // Triangle
    triN1.x=vertices[auxNodeId[0]*3+0];    triN1.y=vertices[auxNodeId[0]*3+1];    triN1.z=vertices[auxNodeId[0]*3+2];
    triN2.x=vertices[auxNodeId[2]*3+0];    triN2.y=vertices[auxNodeId[2]*3+1];    triN2.z=vertices[auxNodeId[2]*3+2];
    triN3.x=vertices[auxNodeId[3]*3+0];    triN3.y=vertices[auxNodeId[3]*3+1];    triN3.z=vertices[auxNodeId[3]*3+2];
    //compute
    l3=moller_trumbore(ray1,ray2,triN1,triN2,triN3,safetyEpsilon);
    // Triangle
    triN1.x=vertices[auxNodeId[1]*3+0];    triN1.y=vertices[auxNodeId[1]*3+1];    triN1.z=vertices[auxNodeId[1]*3+2];
    triN2.x=vertices[auxNodeId[2]*3+0];    triN2.y=vertices[auxNodeId[2]*3+1];    triN2.z=vertices[auxNodeId[2]*3+2];
    triN3.x=vertices[auxNodeId[3]*3+0];    triN3.y=vertices[auxNodeId[3]*3+1];    triN3.z=vertices[auxNodeId[3]*3+2];
    //compute
    l4=moller_trumbore(ray1,ray2,triN1,triN2,triN3,safetyEpsilon);
    
    //dump
    if(!computelenght){
        return (l1!=0.0)|(l2!=0.0)|(l3!=0.0)|(l4!=0.0);
    }else{
        //fuck branches, but what can I do ....
        if ((l1==0.0)&&(l2==0.0)&&(l3==0.0)&&(l4==0.0)){
            t[0]=0.0;t[1]=0.0;t[2]=0.0;t[3]=0.0;
            return false;
        }else{
            t[0]=l1;t[1]=l2;t[2]=l3;t[3]=l4;
            // find which one is the intersection
            return true;
        }
    }
}


bool rayBoxIntersect(const vec3d ray1, const vec3d ray2,const vec3d nodemin, const vec3d nodemax){
    vec3 direction;
    direction.x=ray2.x-ray1.x;
    direction.y=ray2.y-ray1.y;
    direction.z=ray2.z-ray1.z;
    
    float tmin,tymin,tzmin;
    float tmax,tymax,tzmax;
    if (direction.x >= 0){
        tmin = (nodemin.x - ray1.x) / direction.x;
        tmax = (nodemax.x - ray1.x) / direction.x;
        
    }else{
        tmin = (nodemax.x - ray1.x) / direction.x;
        tmax = (nodemin.x - ray1.x) / direction.x;
    }
    
    if (direction.y >= 0){
        tymin = (nodemin.y - ray1.y) / direction.y;
        tymax = (nodemax.y - ray1.y) / direction.y;
    }else{
        tymin = (nodemax.y - ray1.y) / direction.y;
        tymax = (nodemin.y - ray1.y) / direction.y;
    }
    
    if ( (tmin > tymax) || (tymin > tmax) ){
        return false;
    }
    
    if (tymin > tmin){
        tmin = tymin;
    }
    
    if (tymax < tmax){
        tmax = tymax;
    }
    
    if (direction.z >= 0){
        tzmin = (nodemin.z - ray1.z) / direction.z;
        tzmax = (nodemax.z - ray1.z) / direction.z;
    }else{
        tzmin = (nodemax.z - ray1.z) / direction.z;
        tzmax = (nodemin.z - ray1.z) / direction.z;
    }
    
    
    if ((tmin > tzmax) || (tzmin > tmax)){
        return false;
    }
// If we wanted the ts as output
// if (tzmin > tmin){
//     tmin = tzmin;
// }
//
// if (tzmax < tmax){
//     tmax = tzmax;
// }
    
    return true;
}
/*********************************************************************
 ******Fucntion to detect the first triangle to expand the graph******
 ********************************************************************/

void initXrays(const unsigned long* elements, const double* vertices,
        const unsigned long *boundary,const unsigned long nboundary,
        long * d_res, Geometry geo,
        const vec3d source,const vec3d deltaU,const vec3d deltaV,const vec3d uvOrigin,const vec3d nodemin,const vec3d nodemax,
        unsigned int x,unsigned int y){
    
    
    unsigned long  idx =  x  * geo.nDetecV + y;
    if ((x>= (unsigned long )geo.nDetecU) | (y>= (unsigned long )geo.nDetecV))
        return;
    
    // Create ray
    unsigned int pixelV =(unsigned int)geo.nDetecV- y-1;
    unsigned int pixelU =(unsigned int) x;
    vec3d det;
    
    det.x=(uvOrigin.x+pixelU*deltaU.x+pixelV*deltaV.x);
    det.y=(uvOrigin.y+pixelU*deltaU.y+pixelV*deltaV.y);
    det.z=(uvOrigin.z+pixelU*deltaU.z+pixelV*deltaV.z);
    
    bool crossBound=rayBoxIntersect(source, det, nodemin,nodemax);
    if (!crossBound){
        d_res[idx]=-1.0f;
        return;
    }
    
    
    
    // Check intersection with boundary
    unsigned long notintersect=nboundary;
    float t[4];
    float t1,tinter=10000.0f;
    float safetyEpsilon=0.0000001f;
    unsigned long crossingID=0;
    while(notintersect==nboundary){
        notintersect=0;
        for(unsigned long i=0 ;i<nboundary;i++){
            tetraLineIntersect(elements,vertices,source,det,boundary[i],t,true,safetyEpsilon);
            
            if (nnz(t)==0){
                notintersect++;
            }else{
                
                t1=min4nz(t);
                
                if (t1<tinter){
                    
                    tinter=t1;
                    crossingID=i;
                }
            }
        }
        safetyEpsilon=safetyEpsilon*10;
    }
    d_res[idx]=(long)crossingID;
    
    // Should I put the kernels together, or separate? TODO
    
}
/*********************************************************************
 ******************The mein projection fucntion **********************
 ********************************************************************/

void graphBackproject(const unsigned long *elements, const double *vertices,const unsigned long *boundary,const long *neighbours, const float * d_projection,const long *d_init, float * d_res, Geometry geo,
        vec3d source, vec3d deltaU, vec3d deltaV, vec3d uvOrigin, unsigned int x,unsigned int y){
    
    unsigned long  idx =  x  * geo.nDetecV + y;
    if ((x>= (unsigned long )geo.nDetecU) | (y>= (unsigned long )geo.nDetecV))
        return;
    
    // Read initial position.
    // Is having this here going to speed up because the below maths can be done while waiting to read?
    long current_element=(long)d_init[idx];
    long previous_element;
    long aux_element;
    // Create ray
    unsigned int pixelV =(unsigned int)geo.nDetecV- y-1;
    unsigned int pixelU =(unsigned int) x;
    vec3d det;
    
    det.x=(uvOrigin.x+pixelU*deltaU.x+pixelV*deltaV.x);
    det.y=(uvOrigin.y+pixelU*deltaU.y+pixelV*deltaV.y);
    det.z=(uvOrigin.z+pixelU*deltaU.z+pixelV*deltaV.z);
    
    if (current_element==-1){
        //no need to do stuff
//         d_res[idx]=0.0f;
        return;
    }
    
    float result=0.0f;
    
    float length,t1,t2;
    float t[4];
    int indM;
    bool isIntersect;
    
    isIntersect=tetraLineIntersect(elements,vertices,source,det,boundary[current_element],t,true,0.0);
    
    t2=max4(t,&indM);
    t1=min4nz(t);
//      mexPrintf("%.16f %.16f\n",t2,t1);
//     mexPrintf("%.16f %.16f %.16f\n",source.x,source.y,source.z);
//     mexPrintf("%.16f %.16f %.16f\n",det.x,det.y,det.z);
    
    
    vec3d direction,p1,p2;
    direction.x=det.x-source.x;     direction.y=det.y-source.y;     direction.z=det.z-source.z;
    p2.x=direction.x* (t2);  p2.y=direction.y* (t2); p2.z=direction.z* (t2);
    p1.x=direction.x* (t1);  p1.y=direction.y* (t1); p1.z=direction.z* (t1);
    
    length=sqrt((p2.x-p1.x)*(p2.x-p1.x)+(p2.y-p1.y)*(p2.y-p1.y)+(p2.z-p1.z)*(p2.z-p1.z));
    
    
    
//     result=d_image[boundary[current_element]]*length;
    d_res[boundary[current_element]]+=d_projection[idx]*length;
    if(t1==t2){
        
        aux_element=neighbours[boundary[current_element]*4+indM];
        if(aux_element==-1){
            int auxind;
            for(int i=0;i<4;i++){
                if(indM!=i && t[i]==t1){
                    auxind=i;
                }
            }
            indM=auxind;
        }
    }
    
    
    previous_element=boundary[current_element];
    current_element=neighbours[boundary[current_element]*4+indM];
    if (current_element==-1){
        d_res[idx]=result;
        return;
    }
    
    float sumt;
    unsigned long c=0;
    float safeEpsilon=0.00001f;
    bool noNeighbours=false;
    while(!noNeighbours && c<15000){
        c++;
        // get instersection and lengths.
        isIntersect=tetraLineIntersect(elements,vertices,source,det,(unsigned int)current_element,t,true,0.0f);
        while(!isIntersect){
            isIntersect=tetraLineIntersect(elements,vertices,source,det,(unsigned int)current_element,t,true,safeEpsilon);
            if (nnz(t)<=1){
                isIntersect=false;
                safeEpsilon*=10;
            }
        }
        
        t2=max4(t,&indM);
        t1=min4nz(t);
//         mexPrintf("%u %.16f %.16f\n",(unsigned int)current_element,t2,t1);
//         mexPrintf("%.16f \n",(t2-t1));
        if (fabsf(t2-t1)<0.00000001){
            t2=t1;
            t[indM]=t1;
//             mexPrintf("hello! ");
        }
        sumt=0;
        for(int i=0;i<4;i++){
            sumt+=t[i];
        }
        
        if (sumt!=0.0){
            
            p2.x=direction.x* (t2);  p2.y=direction.y* (t2); p2.z=direction.z* (t2);
            p1.x=direction.x* (t1);  p1.y=direction.y* (t1); p1.z=direction.z* (t1);
            length=sqrt((p2.x-p1.x)*(p2.x-p1.x)+(p2.y-p1.y)*(p2.y-p1.y)+(p2.z-p1.z)*(p2.z-p1.z));
            // if (t1==t2); skip following line? timetest
//             result+=d_image[current_element]*length;
            d_res[current_element]+=d_projection[idx]*length;
            if(t1==t2){
                
                aux_element=neighbours[current_element*4+indM];
                if(aux_element==previous_element){
                    int auxind;
                    for(int i=0;i<4;i++){
                        if(indM!=i && t[i]==t1){
                            auxind=i;
//                            mexPrintf("hello! ");
                        }
                    }
                    indM=auxind;
                }
            }
            previous_element=current_element;
            current_element=neighbours[current_element*4+indM];
//             mexPrintf("%ld\n",current_element);
            if (current_element==-1){
//                 d_res[idx]=result;
                return;
            }
            continue;
        }
        noNeighbours=true;
    }//endwhile
//     d_res[idx]=-1.0;
    return;
}
/*********************************************************************
 *********************** Main fucntion ************************
 ********************************************************************/
void graphBackwardRay_CPU(float const * const  projections,  Geometry geo,
        const double * angles,const unsigned int nangles,
        const double* nodes,const unsigned long nnodes,
        const unsigned long* elements,const unsigned long nelements,
        const long* neighbours,const unsigned long nneighbours,
        const unsigned long* boundary,const unsigned long nboundary,
        float * result){
    float time;
    float timecopy, timekernel;
    
    
    
    size_t num_bytes_img  = nelements*sizeof(float);
    size_t num_bytes_proj = geo.nDetecU*geo.nDetecV * sizeof(float);
    
    float * d_res;
    long * d_init;
    d_init=(long*)malloc(geo.nDetecU*geo.nDetecV * sizeof(long));
    d_res=(float*)malloc(num_bytes_img);
    for(int i = 0; i < nelements; i++){
        d_res[i] = 0.;
    }
    
    
    if (DEBUG_TIME){
        
    }
    
    vec3d source2;
    vec3d source,deltaU, deltaV, uvOrigin;
    
    
    vec3d nodemin, nodemax;
    nodemin.x=nodes[0];
    nodemin.y=nodes[1];
    nodemin.z=nodes[2];
    nodemax.x=nodes[0];
    nodemax.y=nodes[1];
    nodemax.z=nodes[2];
    
    for(unsigned long i=1;i<nnodes;i++){
        nodemin.x=(nodes[i*3+0]<nodemin.x)?nodes[i*3+0]:nodemin.x;
        nodemin.y=(nodes[i*3+1]<nodemin.y)?nodes[i*3+1]:nodemin.y;
        nodemin.z=(nodes[i*3+2]<nodemin.z)?nodes[i*3+2]:nodemin.z;
        nodemax.x=(nodes[i*3+0]>nodemax.x)?nodes[i*3+0]:nodemax.x;
        nodemax.y=(nodes[i*3+1]>nodemax.y)?nodes[i*3+1]:nodemax.y;
        nodemax.z=(nodes[i*3+2]>nodemax.z)?nodes[i*3+2]:nodemax.z;
    }
    
    for (unsigned int i=0;i<nangles;i++){
        geo.alpha=angles[i*3];
        geo.theta=angles[i*3+1];
        geo.psi  =angles[i*3+2];
        
// //
        unsigned int initx=0;
        unsigned int inity=0;
        unsigned int niterx=geo.nDetecU+initx; //geo.nDetecU;
        unsigned int nitery=geo.nDetecV+inity; //geo.nDetecV;
//         unsigned int initx=227-1;
//         unsigned int inity=148-1;
//         unsigned int niterx=1+initx; //geo.nDetecU;
//         unsigned int nitery=1+inity; //geo.nDetecV;
        computeGeomtricParams(geo, &source,&deltaU, &deltaV,&uvOrigin,i);
        source2.x=source.x;source2.y=source.y;source2.z=source.z;
        for(unsigned int x=initx;x<(unsigned int)niterx;x++){
            for(unsigned int y=inity;y<(unsigned int)nitery;y++){
                initXrays(elements,nodes,boundary,nboundary, d_init, geo, source2,deltaU, deltaV,uvOrigin,nodemin,nodemax,x,y);
                
                graphBackproject(elements,nodes,boundary,neighbours,&projections[geo.nDetecU*geo.nDetecV*i],d_init,d_res, geo,source2,deltaU,deltaV,uvOrigin,x,y);
                
            }
        }
//         mexPrintf("%f\n",d_res[226*geo.nDetecU+147]);
       
    }
     memcpy(result, d_res, num_bytes_img);
    
    
    return;
    
    
}
//
//
// TODO: quite a lot of geometric transforms.
void computeGeomtricParams(const Geometry geo,vec3d * source, vec3d* deltaU, vec3d* deltaV, vec3d* originUV,unsigned int idxAngle){
    
    vec3d auxOriginUV;
    vec3d auxDeltaU;
    vec3d auxDeltaV;
    auxOriginUV.x=-(geo.DSD[idxAngle]-geo.DSO[idxAngle]);
    // top left
    auxOriginUV.y=-geo.sDetecU/2+/*half a pixel*/geo.dDetecU/2;
    auxOriginUV.z=geo.sDetecV/2-/*half a pixel*/geo.dDetecV/2;
    
    //Offset of the detector
    auxOriginUV.y=auxOriginUV.y+geo.offDetecU[idxAngle];
    auxOriginUV.z=auxOriginUV.z+geo.offDetecV[idxAngle];
    
    // Change in U
    auxDeltaU.x=auxOriginUV.x;
    auxDeltaU.y=auxOriginUV.y+geo.dDetecU;
    auxDeltaU.z=auxOriginUV.z;
    //Change in V
    auxDeltaV.x=auxOriginUV.x;
    auxDeltaV.y=auxOriginUV.y;
    auxDeltaV.z=auxOriginUV.z-geo.dDetecV;
    
    vec3d auxSource;
    auxSource.x=geo.DSO[idxAngle];
    auxSource.y=0;
    auxSource.z=0;
    
    // rotate around axis.
    eulerZYZ(geo,&auxOriginUV);
    eulerZYZ(geo,&auxDeltaU);
    eulerZYZ(geo,&auxDeltaV);
    eulerZYZ(geo,&auxSource);
    
    
    // Offset image (instead of offseting image, -offset everything else)
    auxOriginUV.x  =auxOriginUV.x-geo.offOrigX[idxAngle];     auxOriginUV.y  =auxOriginUV.y-geo.offOrigY[idxAngle];     auxOriginUV.z  =auxOriginUV.z-geo.offOrigZ[idxAngle];
    auxDeltaU.x=auxDeltaU.x-geo.offOrigX[idxAngle];           auxDeltaU.y=auxDeltaU.y-geo.offOrigY[idxAngle];           auxDeltaU.z=auxDeltaU.z-geo.offOrigZ[idxAngle];
    auxDeltaV.x=auxDeltaV.x-geo.offOrigX[idxAngle];           auxDeltaV.y=auxDeltaV.y-geo.offOrigY[idxAngle];           auxDeltaV.z=auxDeltaV.z-geo.offOrigZ[idxAngle];
    auxSource.x=auxSource.x-geo.offOrigX[idxAngle];           auxSource.y=auxSource.y-geo.offOrigY[idxAngle];           auxSource.z=auxSource.z-geo.offOrigZ[idxAngle];
    
    auxDeltaU.x=auxDeltaU.x-auxOriginUV.x;  auxDeltaU.y=auxDeltaU.y-auxOriginUV.y; auxDeltaU.z=auxDeltaU.z-auxOriginUV.z;
    auxDeltaV.x=auxDeltaV.x-auxOriginUV.x;  auxDeltaV.y=auxDeltaV.y-auxOriginUV.y; auxDeltaV.z=auxDeltaV.z-auxOriginUV.z;
    
    *originUV=auxOriginUV;
    *deltaU=auxDeltaU;
    *deltaV=auxDeltaV;
    *source=auxSource;
    
    return;
}

void eulerZYZ(Geometry geo,  vec3d* point){
    vec3d auxPoint;
    auxPoint.x=point->x;
    auxPoint.y=point->y;
    auxPoint.z=point->z;
    
    point->x=(+cos(geo.alpha)*cos(geo.theta)*cos(geo.psi)-sin(geo.alpha)*sin(geo.psi))*auxPoint.x+
            (-cos(geo.alpha)*cos(geo.theta)*sin(geo.psi)-sin(geo.alpha)*cos(geo.psi))*auxPoint.y+
            cos(geo.alpha)*sin(geo.theta)*auxPoint.z;
    
    point->y=(+sin(geo.alpha)*cos(geo.theta)*cos(geo.psi)+cos(geo.alpha)*sin(geo.psi))*auxPoint.x+
            (-sin(geo.alpha)*cos(geo.theta)*sin(geo.psi)+cos(geo.alpha)*cos(geo.psi))*auxPoint.y+
            sin(geo.alpha)*sin(geo.theta)*auxPoint.z;
    
    point->z=-sin(geo.theta)*cos(geo.psi)*auxPoint.x+
            sin(geo.theta)*sin(geo.psi)*auxPoint.y+
            cos(geo.theta)*auxPoint.z;
    
    
    
    
}
