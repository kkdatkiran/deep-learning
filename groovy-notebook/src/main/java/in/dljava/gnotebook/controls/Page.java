package in.dljava.gnotebook.controls;

import in.dljava.gnotebook.book.storage.BookManager;
import in.dljava.gnotebook.book.storage.BookPage;
import in.dljava.gnotebook.book.storage.BookPage.BookPageType;
import javafx.beans.property.SimpleBooleanProperty;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.scene.control.Button;
import javafx.scene.control.TextArea;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;

public class Page extends VBox { // NOSONAR

	private final SimpleBooleanProperty visibility = new SimpleBooleanProperty(false);

	private final BookPage bookPage;
	private final BookManager bookManager;

	private final VBox verticalBar = new VBox();
	private final VBox textContainer = new VBox();

	private final SimpleIntegerProperty count = new SimpleIntegerProperty(180);

	public Page(BookManager bookManager, BookPage bookPage) {

		this.bookManager = bookManager;
		this.bookPage = bookPage;

		this.bookPage.getBookPageTypeProperty().addListener((ov, o, n) -> {
			this.generateBar(n);
			this.generateCenterContainer(n, this.bookPage.getContentProperty().get(),
					this.bookPage.getOutputProperty().get());
		});

		this.setOnMouseEntered(e -> visibility.set(true));
		this.setOnMouseExited(e -> visibility.set(false));

		AddBlockBar addBlockBar = new AddBlockBar(this.bookManager, this.bookPage.getId());
		addBlockBar.getVisibilityProperty().bindBidirectional(visibility);
		this.getChildren().add(addBlockBar);

		verticalBar.visibleProperty().bind(visibility);
		HBox barAndTextContainer = new HBox(verticalBar, textContainer);
		SimpleDoubleProperty widthProperty = new SimpleDoubleProperty(this.getWidth());
		this.widthProperty().addListener((ob, n, o) -> widthProperty.set(o.doubleValue() - 50d));

		textContainer.prefWidthProperty().bind(widthProperty);
		barAndTextContainer.prefWidthProperty().bind(this.prefWidthProperty());

		this.getChildren().add(barAndTextContainer);

		this.generateBar(bookPage.getBookPageTypeProperty().get());
		this.generateCenterContainer(bookPage.getBookPageTypeProperty().get(), bookPage.getContentProperty().get(),
				bookPage.getOutputProperty().get());

		this.getStyleClass().add("page");

	}

	private void generateCenterContainer(BookPageType type, String content, String output) {

		this.textContainer.getChildren().clear();

		if (type == BookPageType.MARKDOWN) {

			this.textContainer.getChildren().add(new Text(content));
		} else {

			TextArea textArea = new TextArea();
			textArea.textProperty().bindBidirectional(bookPage.getContentProperty());
			this.textContainer.getChildren().add(textArea);
			this.textContainer.getChildren().add(new Text(output));

			textArea.prefHeightProperty().bindBidirectional(count);
			textArea.minHeightProperty().bindBidirectional(count);
			textArea.scrollTopProperty().addListener((ov, oldVal, newVal) -> {
				if (newVal.intValue() > 0) {
					count.setValue(count.get() + newVal.intValue() + 28);
				}
			});
		}
	}

	private void generateBar(BookPageType type) {

		var children = this.verticalBar.getChildren();

		children.clear();

		var runButton = new Button("R");
		runButton.setOnAction(a -> bookManager.runPage(bookPage.getId()));
		children.add(runButton);

		var deleteButton = new Button("D");
		deleteButton.setOnAction(a -> bookManager.deleteBookPage(this.bookPage.getId()));
		children.add(deleteButton);
	}

}
