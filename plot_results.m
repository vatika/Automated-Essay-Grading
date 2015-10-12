a = [ 0.790134505334, 0.642857955445, 0.643522406954, 0.750886847833, 0.692496251218 ];
b = [ 0.817330565666, 0.693137082379, 0.627502607493, 0.775082249743, 0.657474984478 ];
figure, plot([1,2,3,4,5],a,'ro--','MarkerSize',10), hold on;
plot([1,2,3,4,5],b,'bd--','MarkerSize',10), axis([0,6,-1,1]),xlabel('essay set #'),ylabel('kappa value'), hold off;
legend('linear regression','support vector regression', 'Location', 'SouthEast');