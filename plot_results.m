a  = [0.790134505334, 0.642857955445, 0.643522406954, 0.750886847833, 0.692496251218];
b  = [0.817330565666,0.693137082379, 0.627502607493, 0.775082249743, 0.657474984478];
c  = [0.733384200813,0.677537216387, 0.657046522605, 0.751634408086, 0.674324581641];
d  = [0.78137228711' 0.685843977459, 0.640135434635, 0.761360207999, 0.638876379903];
e  = [0.816265180433,0.672892228774, 0.64641240482, 0.765672271065, 0.673981182921];
f  = [0.714811955651,0.518820668951, 0.581295967478, 0.657223719396, 0.579141163064];

figure, plot([1,2,3,4,5],a,'ro:','MarkerSize',10), hold on;
plot([1,2,3,4,5],b,'bd--','MarkerSize',5);
plot([1,2,3,4,5],c,'kx--','MarkerSize',5);
plot([1,2,3,4,5],d,'m+-','MarkerSize',5);
plot([1,2,3,4,5],e,'rp-.','MarkerSize',5);
plot([1,2,3,4,5],f,'kh-','MarkerSize',5), axis([0,6,0.5,1]),xlabel('essay set #'),ylabel('kappa value'), hold off;
legend('linear regression','support vector regression','logistic regression', 'support vector machine' ,'kernel ridge regression', 'decision tree classifier', 'Location', 'SouthEast');
