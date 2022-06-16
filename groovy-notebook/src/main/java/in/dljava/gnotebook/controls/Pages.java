package in.dljava.gnotebook.controls;

import in.dljava.gnotebook.book.storage.BookManager;
import in.dljava.gnotebook.book.storage.BookPage;
import javafx.collections.ListChangeListener;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.FlowPane;

public class Pages extends ScrollPane { // NOSONAR - This class can
										// have as many parents it
										// want.

	private final BookManager bookManager;

	private final FlowPane container = new FlowPane();

	private final AddBlockBar lastBar;

	public Pages(BookManager bookManager) {

		this.setId("pages");

		this.bookManager = bookManager;
		this.setVbarPolicy(ScrollBarPolicy.AS_NEEDED);
		this.setHbarPolicy(ScrollBarPolicy.NEVER);

		this.lastBar = new AddBlockBar(bookManager, -1);
		this.lastBar.setId("lastBar");

		this.container.getChildren().add(lastBar);
		this.container.prefWidthProperty().bind(this.widthProperty());
		this.lastBar.prefWidthProperty().bind(this.container.widthProperty());
		this.container.getStyleClass().add("pagesContainer");

		this.setContent(container);
		this.bookManager.getPagesProperty().addListener(this::listChanged);
		
		this.getStyleClass().add("pages");
	}

	private void listChanged(ListChangeListener.Change<? extends BookPage> page) {

		System.out.println(page);
	}

}
