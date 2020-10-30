<template>
    <div id="contextual-menu">
      <div v-for="(editor, index) in editors" :key="index">
        <VerticalMenuButton
          :label="editor.name + (editor.saved ? '' : '*')"
          :onClick="() => switchEditor({ editorType, index})"
          :isSelected="editorType === currEditorType && index === currEditorIndex">
          <SaveEditorButton :index="index" v-if="save"/>
        </VerticalMenuButton>
      </div>
      <VerticalMenuButton
        :label="'+'"
        :onClick="this.createNewEditor"
        :isSelected="false"
      />
    </div>
</template>

<script lang="ts">
import { Component, Prop, Vue } from 'vue-property-decorator';
import VerticalMenuButton from '@/components/buttons/VerticalMenuButton.vue';
import { mapGetters, mapMutations } from 'vuex';
import EditorType from '@/EditorType';
import { Getter, Mutation } from 'vuex-class';
import SaveEditorButton from '@/components/buttons/SaveEditorButton.vue';
import { EditorModel } from '@/store/editors/types';

@Component({
  components: { SaveEditorButton, VerticalMenuButton },
  computed: mapGetters([
    'currEditorType',
    'currEditorIndex',
  ]),
  methods: mapMutations(['switchEditor']),
})
export default class NavbarContextualMenu extends Vue {
  @Prop({ required: true }) readonly editors!: EditorModel[];
  @Prop({ required: true }) readonly editorType!: EditorType;
  @Prop() readonly save?: boolean;
  @Getter('editorNames') editorNames!: Set<string>;
  @Mutation('newEditor') newEditor!: (arg0: { editorType: EditorType; name: string}) => void;

  private createNewEditor(): void {
    let isNameUnique = false;
    while (!isNameUnique) {
      const name = prompt('Please enter a unique name for the editor');

      // Name is null if cancelled
      if (name === null) break;

      // Loop until unique non-empty name has been entered
      if (name !== '' && !this.editorNames.has(name)) {
        isNameUnique = true;
        this.newEditor({ editorType: this.editorType, name });
      }
    }
  }
}

</script>

<style scoped>
  #contextual-menu {
    border: 1px solid var(--grey);
  }
</style>
