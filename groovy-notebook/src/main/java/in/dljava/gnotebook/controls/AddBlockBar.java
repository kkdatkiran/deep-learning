package in.dljava.gnotebook.controls;

import in.dljava.gnotebook.book.storage.BookManager;
import in.dljava.gnotebook.book.storage.BookPage.BookPageType;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.layout.HBox;

public class AddBlockBar extends HBox { // NOSONAR - this class needs the inheritance levels.

	private final SimpleBooleanProperty visibility = new SimpleBooleanProperty(false);

	public AddBlockBar(BookManager bookManager, String id) {

		Button addCode = new Button("+ Code");
		Button addMKD = new Button("+ Markdown");

		addCode.getStyleClass().add("addBarButton");
		addMKD.getStyleClass().add("addBarButton");
		this.getStyleClass().add("addBar");

		addCode.visibleProperty().bind(visibility);
		addMKD.visibleProperty().bind(visibility);

		addCode.setOnAction(e -> bookManager.addBookPage(BookPageType.CODE, id));
		addMKD.setOnAction(e -> bookManager.addBookPage(BookPageType.MARKDOWN, id));

		this.setOnMouseEntered(e -> visibility.setValue(true));
		this.setOnMouseExited(e -> visibility.setValue(false));

		Label t = new Label("/");
		t.visibleProperty().bind(visibility);
		t.getStyleClass().add("addBarText");

		this.getChildren().addAll(addCode, t, addMKD);
	}

	public SimpleBooleanProperty getVisibilityProperty() {
		return this.visibility;
	}
}
