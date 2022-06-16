package in.dljava.gnotebook.controls;

import javafx.beans.property.StringProperty;
import javafx.scene.control.Label;
import javafx.scene.layout.HBox;

public class StatusBar extends HBox { // NOSONAR - This class can have as many parents it want.

	private Label text;

	public StatusBar() {

		text = new Label();
		text.setId("statusBarText");
		text.setText("initial");

		this.getChildren().add(text);
		this.setId("statusBar");
	}

	public StringProperty textProperty() {
		return text.textProperty();
	}

}