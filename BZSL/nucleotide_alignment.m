clear all
clc

load('C:\Users\selsh\repos\Fine-Grained-ZSL-with-DNA\data\INSECT\res101.mat')
load('C:\Users\selsh\repos\Fine-Grained-ZSL-with-DNA\data\INSECT\att_splits.mat');
nucleotides=nucleotides; 
n=numel(nucleotides);
for i=1:n
    tmp=nucleotides{i};
    
    c=1; ind_s=[];
    tmpi=tmp(c);
    while strcmp(tmpi,'N')==1 | strcmp(tmpi,'-')==1
        ind_s=[ind_s c];
        c=c+1;
        tmpi=tmp(c);
    end
    
     c=length(tmp); ind_e=[];
    tmpi=tmp(c);
    while strcmp(tmpi,'N')==1 | strcmp(tmpi,'-')==1
        ind_e=[ind_e c];
        c=c-1;
        tmpi=tmp(c);
    end
    tmp([ind_s ind_e])=[];
nucleotides{i}=tmp;
end

for i=1:n
lnt(i)=length(nucleotides{i});
end



xtrain=nucleotides(trainval_loc);
n=numel(xtrain);
lnt=zeros(n,1);
lntd=zeros(n,1);
c=1;
for i=1:n
    lnt(i)=length(strfind(xtrain{i},'A'))+length(strfind(xtrain{i},'G'))+length(strfind(xtrain{i},'C'))+length(strfind(xtrain{i},'T'));
    lntd(i)=length(strfind(xtrain{i},'-'))+length(strfind(xtrain{i},'N'));
    if lnt(i)==658 & lntd(i)==0
    filtered{c}=xtrain{i};
    c=c+1
    end
end
proto=seqconsensus(filtered,'gaps','noflanks');


n=numel(nucleotides);

for i=1:n
    clear tmp;
    tmp{1}=proto; tmp{2}=nucleotides{i}; tmp{3}=proto;
    i
    alg=multialign(tmp,'existingGapAdjust',false);
    nucleotides_aligned{i}=alg(2,:);
end