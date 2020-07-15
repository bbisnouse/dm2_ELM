package project1;



import no.uib.cipr.matrix.NotConvergedException;

public class Main {
	public static void main(String[] args) throws NotConvergedException {
		// TODO Auto-generated method stub
		//h = 20
		Elm ds = new Elm(20);
		ds.train("train.txt");	
		ds.test("test.txt");
		System.out.println("\nTraining_Accuracy = "+ds.getTrainingAccuracy());
	}

}
